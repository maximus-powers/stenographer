"""The event tap: where real keystrokes enter the system.

This process holds an *active* tap, which means every keystroke on the machine passes
through the callback below before reaching the application that will receive it. That is
what makes chording possible and it is also what makes this the most dangerous code in
the project: if the callback stalls, keyboard input freezes for every application until
macOS notices, and its recovery from that is not reliable. So nothing here blocks. No
model, no Accessibility call, no synchronous file write, no network.

Three failure modes are handled explicitly because each one is silent otherwise:

* The tap can be switched off by the system when a callback takes too long. That arrives
  as an event of its own and the tap has to be turned back on, with the detector reset
  because keystrokes were lost while it was off.
* Accessibility permission can be revoked while running, which on current macOS hangs
  input with no notification at all. A periodic probe catches it and stands the tap down.
* Secure Input, used by password fields, silently stops keyboard events reaching any tap.
  The detector must not be left holding a half-finished chord when that happens.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import Quartz
from CoreFoundation import (
    CFMachPortCreateRunLoopSource,
    CFRunLoopAddSource,
    CFRunLoopGetCurrent,
    CFRunLoopRunInMode,
    CFRunLoopStop,
    kCFRunLoopCommonModes,
    kCFRunLoopDefaultMode,
)

from stenographer.keyboard.events import KeyEvent, Kind
from stenographer.keyboard.inject import is_ours
from stenographer.keyboard.layout import KeyboardLayout

KEYBOARD_MASK = (
    Quartz.CGEventMaskBit(Quartz.kCGEventKeyDown)
    | Quartz.CGEventMaskBit(Quartz.kCGEventKeyUp)
    | Quartz.CGEventMaskBit(Quartz.kCGEventFlagsChanged)
)

# The system reports these two through the callback in place of a real event.
DISABLED_EVENTS = {
    Quartz.kCGEventTapDisabledByTimeout,
    Quartz.kCGEventTapDisabledByUserInput,
}


def accessibility_trusted(prompt: bool = False) -> bool:
    """Is this process allowed to observe and post events?

    The permission belongs to whichever process is held responsible, which for a script
    run from a terminal is the terminal application, not Python. Passing `prompt` asks
    macOS to show the dialog that offers to open the right settings pane.
    """
    from ApplicationServices import (
        AXIsProcessTrustedWithOptions,
        kAXTrustedCheckOptionPrompt,
    )

    options = {kAXTrustedCheckOptionPrompt: True} if prompt else None
    return bool(AXIsProcessTrustedWithOptions(options))


def secure_input_enabled() -> bool:
    """Is a password field taking the keyboard?

    While this is true no event tap receives key events at all, so there is nothing to be
    done but notice and wait. Occasional false positives from the login window are
    expected on current macOS.
    """
    import ctypes
    import ctypes.util

    path = ctypes.util.find_library("Carbon")
    if not path:
        return False
    try:
        carbon = ctypes.CDLL(path)
        return bool(carbon.IsSecureEventInputEnabled())
    except (OSError, AttributeError):
        return False


@dataclass
class TapStats:
    events: int = 0
    reenabled: int = 0
    max_callback_ms: float = 0.0
    total_callback_ms: float = 0.0

    @property
    def mean_callback_ms(self) -> float:
        return self.total_callback_ms / self.events if self.events else 0.0


class EventTap:
    """Wraps a CGEventTap and drives a handler from its run loop.

    `handler` receives one `KeyEvent` and returns True to let the keystroke through or
    False to swallow it. It must return promptly; see the module docstring.
    """

    def __init__(
        self,
        handler: Callable[[KeyEvent, object], bool],
        *,
        listen_only: bool = False,
        on_timer: Callable[[float], None] | None = None,
        timer_interval: float = 0.005,
        health_interval: float = 5.0,
        on_lost_events: Callable[[], None] | None = None,
    ) -> None:
        self.handler = handler
        self.listen_only = listen_only
        self.on_timer = on_timer
        self.timer_interval = timer_interval
        self.health_interval = health_interval
        self.on_lost_events = on_lost_events
        self.layout = KeyboardLayout.current()
        self.stats = TapStats()
        self.tap = None
        self.running = False
        self._last_health_check = 0.0

    # -- lifecycle ------------------------------------------------------------

    def create(self) -> None:
        option = (
            Quartz.kCGEventTapOptionListenOnly
            if self.listen_only
            else Quartz.kCGEventTapOptionDefault
        )
        self.tap = Quartz.CGEventTapCreate(
            Quartz.kCGSessionEventTap,
            Quartz.kCGHeadInsertEventTap,
            option,
            KEYBOARD_MASK,
            self._callback,
            None,
        )
        if self.tap is None:
            raise PermissionError(
                "could not create an event tap. Grant Accessibility permission to the "
                "application running this process (System Settings > Privacy & Security "
                "> Accessibility), then start it again. Run from a terminal during "
                "development and grant the permission to that terminal."
            )

    def run(self) -> None:
        """Install the tap and process events until stopped.

        The run loop is pumped in short slices rather than entered with `CFRunLoopRun`,
        which would block inside C indefinitely. Python only runs a signal handler between
        its own bytecodes, so with a blocking call there Ctrl-C does nothing and the
        process has to be killed outright. For something sitting between the user and
        their keyboard, being able to stop it is not negotiable.

        Returning to Python each slice also drives the detector's clock: it decides a group
        by timing as much as by keys, so something has to wake it when nothing is pressed.
        This happens on the same thread as the callback, so the two can never race over
        the detector's state.
        """
        if self.tap is None:
            self.create()
        source = CFMachPortCreateRunLoopSource(None, self.tap, 0)
        CFRunLoopAddSource(CFRunLoopGetCurrent(), source, kCFRunLoopCommonModes)
        Quartz.CGEventTapEnable(self.tap, True)
        self.running = True
        self._loop = CFRunLoopGetCurrent()

        try:
            while self.running:
                CFRunLoopRunInMode(kCFRunLoopDefaultMode, self.timer_interval, False)
                if self.on_timer is not None:
                    self.on_timer(time.monotonic() * 1000.0)
                self._check_health()
        finally:
            if self.tap is not None:
                Quartz.CGEventTapEnable(self.tap, False)

    def stop(self) -> None:
        self.running = False
        if self.tap is not None:
            Quartz.CGEventTapEnable(self.tap, False)
        loop = getattr(self, "_loop", None)
        if loop is not None:
            CFRunLoopStop(loop)

    # -- health ---------------------------------------------------------------

    def _check_health(self) -> None:
        if not self.health_interval:
            return
        now = time.monotonic()
        if now - self._last_health_check < self.health_interval:
            return
        self._last_health_check = now
        if self.tap is None:
            return
        if not Quartz.CGEventTapIsEnabled(self.tap):
            # Re-enable rather than recreate: the port is still valid and recreating one
            # while permission is being revoked is what hangs input.
            Quartz.CGEventTapEnable(self.tap, True)
            self.stats.reenabled += 1
            if self.on_lost_events is not None:
                self.on_lost_events()

    # -- the callback ---------------------------------------------------------

    def _callback(self, proxy, event_type, event, _refcon):
        started = time.perf_counter()
        try:
            if event_type in DISABLED_EVENTS:
                # Keystrokes were lost while the tap was off, so whatever the detector was
                # holding is now inconsistent.
                Quartz.CGEventTapEnable(self.tap, True)
                self.stats.reenabled += 1
                if self.on_lost_events is not None:
                    self.on_lost_events()
                return event

            if is_ours(event):
                return event  # our own output, on its way to the application

            key_event = self.to_key_event(event_type, event)
            keep = self.handler(key_event, event)
            return event if keep or self.listen_only else None
        except Exception:
            # A raised exception here would otherwise be swallowed by the callback
            # boundary and leave the keystroke in limbo. Letting it through is always the
            # safer failure: the user sees what they typed.
            import traceback

            traceback.print_exc()
            return event
        finally:
            elapsed = (time.perf_counter() - started) * 1000.0
            self.stats.events += 1
            self.stats.total_callback_ms += elapsed
            self.stats.max_callback_ms = max(self.stats.max_callback_ms, elapsed)

    def to_key_event(self, event_type, event) -> KeyEvent:
        """Convert a CGEvent into the plain record the detector understands."""
        keycode = int(
            Quartz.CGEventGetIntegerValueField(event, Quartz.kCGKeyboardEventKeycode)
        )
        flags = Quartz.CGEventGetFlags(event)
        autorepeat = bool(
            Quartz.CGEventGetIntegerValueField(event, Quartz.kCGKeyboardEventAutorepeat)
        )
        if event_type == Quartz.kCGEventKeyDown:
            kind = Kind.DOWN
        elif event_type == Quartz.kCGEventKeyUp:
            kind = Kind.UP
        else:
            kind = Kind.FLAGS
        return KeyEvent(
            kind=kind,
            t_ms=time.monotonic() * 1000.0,
            key=self.layout.letter(keycode),
            keycode=keycode,
            shift=bool(flags & Quartz.kCGEventFlagMaskShift),
            caps_lock=bool(flags & Quartz.kCGEventFlagMaskAlphaShift),
            autorepeat=autorepeat,
            command=bool(
                flags
                & (
                    Quartz.kCGEventFlagMaskCommand
                    | Quartz.kCGEventFlagMaskControl
                    | Quartz.kCGEventFlagMaskAlternate
                )
            ),
        )
