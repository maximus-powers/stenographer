"""Check whether this machine will actually let stenographer see and send keystrokes.

Worth running first, because the failure is silent. `CGEventTapCreate` returns a valid
tap without Accessibility permission and `CGEventTapIsEnabled` then reports it as
enabled, but no keystroke is ever delivered to it. A build that trusted those two calls
would look like it had started correctly and do nothing at all.

So this does not ask permission questions; it posts a synthetic keystroke and checks
whether the tap receives it. That is the only answer that means anything.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import Quartz
from CoreFoundation import (
    CFMachPortCreateRunLoopSource,
    CFRunLoopAddSource,
    CFRunLoopGetCurrent,
    CFRunLoopRunInMode,
    kCFRunLoopCommonModes,
    kCFRunLoopDefaultMode,
)

from stenographer.keyboard.inject import MARKER, Emitter, is_ours
from stenographer.keyboard.layout import KeyboardLayout
from stenographer.keyboard.tap import (
    KEYBOARD_MASK,
    accessibility_trusted,
    secure_input_enabled,
)

# An unused key code, so a stray delivery cannot be mistaken for the user typing.
PROBE_KEYCODE = 10


def can_create(option) -> bool:
    tap = Quartz.CGEventTapCreate(
        Quartz.kCGSessionEventTap,
        Quartz.kCGHeadInsertEventTap,
        option,
        KEYBOARD_MASK,
        lambda proxy, event_type, event, refcon: event,
        None,
    )
    return tap is not None


def events_are_delivered(timeout_seconds: float = 1.0) -> tuple[int, int]:
    """Post a marked keystroke and count what comes back.

    Returns (events seen, events recognised as ours). Zero seen means either the tap gets
    nothing or posting did nothing; both need the same permission, and both leave the tool
    inert, so they are reported together.
    """
    seen: list[bool] = []

    def callback(proxy, event_type, event, refcon):
        if event_type in (Quartz.kCGEventKeyDown, Quartz.kCGEventKeyUp):
            seen.append(is_ours(event))
        return event

    tap = Quartz.CGEventTapCreate(
        Quartz.kCGSessionEventTap,
        Quartz.kCGHeadInsertEventTap,
        Quartz.kCGEventTapOptionListenOnly,
        KEYBOARD_MASK,
        callback,
        None,
    )
    if tap is None:
        return (0, 0)

    source = CFMachPortCreateRunLoopSource(None, tap, 0)
    CFRunLoopAddSource(CFRunLoopGetCurrent(), source, kCFRunLoopCommonModes)
    Quartz.CGEventTapEnable(tap, True)

    emitter = Emitter()
    for pressed in (True, False):
        event = Quartz.CGEventCreateKeyboardEvent(emitter.source, PROBE_KEYCODE, pressed)
        Quartz.CGEventSetIntegerValueField(event, Quartz.kCGEventSourceUserData, MARKER)
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, event)

    deadline = timeout_seconds
    while deadline > 0 and not seen:
        CFRunLoopRunInMode(kCFRunLoopDefaultMode, 0.02, True)
        deadline -= 0.02

    Quartz.CGEventTapEnable(tap, False)
    return (len(seen), sum(1 for ours in seen if ours))


def main() -> None:
    print("stenographer permission check\n")

    layout = KeyboardLayout.current()
    letters = sum(1 for keycode in layout.to_char if layout.letter(keycode))
    print(f"  keyboard layout      {letters}/26 letters mapped")

    trusted = accessibility_trusted()
    print(f"  Accessibility        {'granted' if trusted else 'NOT granted'}")
    print(f"  Secure Input         {'ON (a password field has the keyboard)' if secure_input_enabled() else 'off'}")

    listen = can_create(Quartz.kCGEventTapOptionListenOnly)
    active = can_create(Quartz.kCGEventTapOptionDefault)
    print(f"  listen-only tap      {'created' if listen else 'refused'}")
    print(f"  active tap           {'created' if active else 'refused'}")

    seen, ours = events_are_delivered()
    print(f"  events delivered     {seen} (of which {ours} recognised as our own)")

    print()
    if seen and active:
        print("  Ready. Chording can capture and inject keystrokes.")
        if not ours:
            print(
                "  Warning: our own events were not recognised by the marker, which means\n"
                "  injected text would be re-read as if the user had typed it."
            )
        return

    print("  Not ready. Nothing will be captured or typed until permission is granted.")
    print()
    if not seen:
        print("  A tap can be created but receives no events, which is what missing")
        print("  Accessibility permission looks like: it fails silently rather than")
        print("  reporting an error.")
    print()
    executable = Path(sys.executable).resolve()
    print("  To fix:")
    print("    1. Open System Settings > Privacy & Security > Accessibility.")
    print("    2. Add the application that runs this process and switch it on.")
    print("       Run from a terminal and that is the terminal application itself,")
    print("       whose permission its child processes inherit. The interpreter here is")
    print(f"       {executable}")
    print("    3. Restart the process and run this check again.")
    print()
    print("  Passing --prompt asks macOS to show the dialog that opens the right pane.")


if __name__ == "__main__":
    if "--prompt" in sys.argv:
        accessibility_trusted(prompt=True)
    main()
