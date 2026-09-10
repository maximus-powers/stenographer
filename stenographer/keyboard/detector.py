"""Deciding whether keys pressed together are a chord or just fast typing.

This is the crux of the whole idea. A fast typist overlaps a large share of consecutive
keystrokes, so "two keys were down at once" is nowhere near enough evidence on its own.
Too loose a test and ordinary typing produces spurious words; too tight and deliberate
chords are missed.

The way out is to stop relying on speed. A chord gets half a second to be formed, and
what marks it out instead is that every one of its keys was down at the same moment. A
typist releases each key about as the next goes down, so ordinary typing never gets three
keys down together however fast it goes, while a deliberate press can be made as slowly
as the hand likes. Two keys are the weak case, since a pair overlapping is exactly what
rollover looks like, so a pair additionally has to be held together long enough to have
been meant.

It is a pure state machine. Every input is a `KeyEvent` or a timer tick, and every output
is an `Action` for the emitter to carry out. Nothing here touches the keyboard, so the
tests drive it with written-out timings, and the thresholds can be retuned by replaying
recorded traces rather than by guessing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from stenographer.keyboard.events import KeyEvent, Kind


class Trigger(Enum):
    """How a chord is signalled."""

    # Hold letters back briefly and decide from timing alone. Costs every plain letter the
    # join window in latency, and relies on the overlap tests below to stay quiet.
    HOLDBACK = "holdback"
    # Let letters through immediately and take them back if a chord turns out to have been
    # formed. No typing latency, at the price of a visible flicker on every chord.
    OPTIMISTIC = "optimistic"
    # Only letters pressed while a designated key is held count as a chord. No false
    # positives and no added latency, but the user has to learn the modifier.
    HOLD_TO_CHORD = "hold_to_chord"


@dataclass(frozen=True)
class DetectorConfig:
    # How long a whole chord gesture may take, from the first key going down to the last
    # coming up. Half a second is enough to place three or four fingers deliberately and
    # still far longer than any single keystroke in ordinary typing. The window is not
    # what keeps ordinary typing out - the simultaneity rule below is - so it is set to
    # what a hand needs rather than to the widest value that would still be safe.
    chord_max_ms: float = 500.0
    # A lone key that is still down after this long is sent on its own. Until then it is
    # held back, because another finger may still be coming to join it, and cutting this
    # short would break up a slowly formed chord.
    #
    # This costs ordinary typing nothing, which is easy to get wrong: a normal keystroke
    # is decided the moment it is released, not when this expires, so the letter appears
    # as the finger lifts. What it does affect is holding a letter key down to repeat it,
    # where the first character waits this long. Prose rarely does that, and the places
    # that do, terminals and editors, are on the deny-list anyway. Lower it if key repeat
    # matters more than a leisurely chord.
    lone_hold_ms: float = 500.0
    # A two-key group needs this much simultaneous hold, measured from the second key
    # going down to the first coming up. Two keys overlapping is the single weakest
    # signal there is, because that is exactly what fast typing looks like, so a pair has
    # to be held together long enough to be deliberate.
    two_key_overlap_ms: float = 60.0
    min_keys: int = 2
    # Laptop keyboards register about six non-modifier keys at once; more than that means
    # ghosting or a rollover burst rather than a chord.
    max_keys: int = 6
    # Require every key in the chord to have been down at one instant. This is what makes
    # a long gesture safe: the time limit alone would swallow whole words of ordinary
    # typing, but rollover never gets three keys down together, so demanding that all of
    # them overlap separates intent from speed without needing a narrow window.
    require_simultaneous: bool = True
    trigger: Trigger = Trigger.HOLDBACK


class Action:
    """Something the emitter should do."""


@dataclass(frozen=True)
class PassThrough(Action):
    """Let the event reach the application untouched."""

    event: KeyEvent


@dataclass(frozen=True)
class Consume(Action):
    """Drop the event; it has been accounted for."""

    event: KeyEvent


@dataclass(frozen=True)
class TypeLetters(Action):
    """Send these held-back letters after all, in the order they were pressed.

    `synthetic_up` says whether the emitter must generate the key releases. It is false
    while the user is still holding a key, so the real release arrives by itself and key
    repeat keeps working.
    """

    events: tuple[KeyEvent, ...]
    synthetic_up: bool = True

    @property
    def letters(self) -> str:
        return "".join(event.key or "" for event in self.events)


@dataclass(frozen=True)
class CommitChord(Action):
    """Ask for a prediction for these letters and type the winning word."""

    letters: str  # sorted and deduplicated: the chord's identity
    press_order: str  # order the keys went down, a weak signal at best
    shift: bool
    caps_lock: bool
    t_ms: float


@dataclass(frozen=True)
class Retract(Action):
    """Remove characters already sent, when an optimistic guess turns into a chord."""

    count: int


@dataclass
class Group:
    """Letters currently being considered as one chord."""

    t0: float
    events: list[KeyEvent] = field(default_factory=list)
    held: set[str] = field(default_factory=set)
    released_at: dict[str, float] = field(default_factory=dict)
    max_simultaneous: int = 0
    emitted: int = 0  # characters already sent through, in optimistic mode

    @property
    def keys(self) -> list[str]:
        return [event.key for event in self.events if event.key]

    @property
    def last_down_ms(self) -> float:
        return self.events[-1].t_ms if self.events else self.t0

    @property
    def released_any(self) -> bool:
        return bool(self.released_at)

    @property
    def shift(self) -> bool:
        return any(event.shift for event in self.events)

    @property
    def caps_lock(self) -> bool:
        return any(event.caps_lock for event in self.events)

    def add(self, event: KeyEvent) -> None:
        self.events.append(event)
        if event.key:
            self.held.add(event.key)
        self.max_simultaneous = max(self.max_simultaneous, len(self.held))

    def release(self, key: str, t_ms: float) -> None:
        self.held.discard(key)
        self.released_at.setdefault(key, t_ms)

    def first_release_ms(self) -> float | None:
        return min(self.released_at.values()) if self.released_at else None


class ChordDetector:
    """Classifies keystrokes as chords or ordinary letters."""

    def __init__(self, config: DetectorConfig | None = None) -> None:
        self.config = config or DetectorConfig()
        self.group: Group | None = None
        # Keys whose press was held back, so whose release must be held back too.
        self.pending_release: set[str] = set()
        self.chord_modifier_down = False
        self.enabled = True

    # -- inputs ---------------------------------------------------------------

    def feed(self, event: KeyEvent) -> list[Action]:
        if not self.enabled:
            return [PassThrough(event)]
        if event.kind is Kind.FLAGS:
            return [PassThrough(event)]
        if not event.is_letter:
            return self._non_letter(event)
        return self._letter_down(event) if event.is_down else self._letter_up(event)

    def on_timer(self, t_ms: float) -> list[Action]:
        """Advance time. Call whenever the time from `next_deadline` has passed."""
        if self.group is None:
            return []
        deadline = self.next_deadline()
        if deadline is None or t_ms < deadline:
            return []
        return self._resolve(t_ms)

    def next_deadline(self) -> float | None:
        """When the open group must be decided, or None if nothing is pending."""
        group = self.group
        config = self.config
        if group is None:
            return None
        if len(group.events) < config.min_keys:
            # Still one key. It is held back in case a second finger is coming, but not
            # past the point where key repeat should have started.
            return group.t0 + config.lone_hold_ms
        # Enough keys for a chord, so wait for the fingers to lift, or for the gesture to
        # run out of time.
        return group.t0 + config.chord_max_ms

    def reset(self) -> list[Action]:
        """Abandon any open group, sending its letters through.

        Called when events were lost, which happens when the tap is disabled by timeout or
        when Secure Input takes the keyboard away in the middle of a press.
        """
        actions: list[Action] = []
        if self.group is not None:
            group, self.group = self.group, None
            actions.extend(self._flush_as_letters(group, synthetic_up=True))
        self.pending_release.clear()
        return actions

    # -- letters --------------------------------------------------------------

    def _letter_down(self, event: KeyEvent) -> list[Action]:
        config = self.config
        key = event.key or ""

        if config.trigger is Trigger.HOLD_TO_CHORD and not self.chord_modifier_down:
            self.pending_release.discard(key)
            return [PassThrough(event)]

        if event.autorepeat:
            # Repeat cannot begin inside the join window, since the system floor is 225 ms,
            # so a repeat belongs either to a group already decided or to a key we never
            # held back.
            if self.group is not None and key in self.group.held:
                return [Consume(event)]
            self.pending_release.discard(key)
            return [PassThrough(event)]

        group = self.group
        if group is None:
            return self._open_group(event)

        if key in group.held:
            # The same key pressed twice without a release means its release was lost.
            return self._resolve(event.t_ms) + self._open_group(event)

        # A key joins only while every key already in the group is still down. That single
        # rule is what lets the gesture take as long as it needs without swallowing
        # ordinary typing: a typist releases each key around the time the next goes down,
        # so the moment anything has been let go the group is closed and a new one starts.
        # It also means the keys of a chord were necessarily all down at one instant.
        elapsed = event.t_ms - group.t0
        if not group.released_any and elapsed <= config.chord_max_ms:
            # A key that pushes the group past `max_keys` still joins it. Rejecting the
            # extra key instead would leave a full-sized group behind and commit it as a
            # chord, when the whole burst is what looks wrong: too many keys at once means
            # a rollover run or a ghosting keyboard, so all of it should be typed out.
            group.add(event)
            return self._hold(event, group)

        return self._resolve(event.t_ms) + self._open_group(event)

    def _letter_up(self, event: KeyEvent) -> list[Action]:
        key = event.key or ""
        group = self.group
        if group is not None and key in group.keys:
            group.release(key, event.t_ms)
            # This release is being consumed here, so it no longer needs holding back.
            self.pending_release.discard(key)
            if not group.held:
                # Every finger is off, so the gesture is complete and can be judged. This
                # is the normal way a group ends, and waiting for it is what allows a
                # chord to be formed slowly. The release itself is still consumed: it
                # belongs to a keystroke the application never saw.
                return [Consume(event), *self._resolve(event.t_ms)]
            return [Consume(event)]
        if key in self.pending_release:
            self.pending_release.discard(key)
            return [Consume(event)]
        return [PassThrough(event)]

    def _non_letter(self, event: KeyEvent) -> list[Action]:
        """Anything that is not a plain letter closes an open group, then passes through."""
        actions: list[Action] = []
        if self.group is not None and event.is_down:
            actions.extend(self._resolve(event.t_ms))
        actions.append(PassThrough(event))
        return actions

    # -- group handling -------------------------------------------------------

    def _open_group(self, event: KeyEvent) -> list[Action]:
        group = Group(t0=event.t_ms)
        group.add(event)
        self.group = group
        return self._hold(event, group)

    def _hold(self, event: KeyEvent, group: Group) -> list[Action]:
        """Either hold the keystroke back or let it through, depending on the strategy."""
        if self.config.trigger is Trigger.OPTIMISTIC:
            group.emitted += 1
            return [PassThrough(event)]
        if event.key:
            self.pending_release.add(event.key)
        return [Consume(event)]

    def _resolve(self, t_ms: float) -> list[Action]:
        group = self.group
        if group is None:
            return []
        self.group = None

        if not self._is_chord(group, t_ms):
            return self._flush_as_letters(group, synthetic_up=not group.held)

        # A chord replaces its keystrokes with a word, so every press it swallowed stays
        # swallowed: keys still held keep their entry in `pending_release` and their real
        # releases are dropped when they arrive, rather than reaching the application.

        actions: list[Action] = []
        if group.emitted:
            actions.append(Retract(group.emitted))
        actions.append(
            CommitChord(
                letters="".join(sorted(set(group.keys))),
                press_order="".join(dict.fromkeys(group.keys)),
                shift=group.shift,
                caps_lock=group.caps_lock,
                t_ms=t_ms,
            )
        )
        return actions

    def _is_chord(self, group: Group, t_ms: float) -> bool:
        config = self.config
        count = len(group.events)
        if count < config.min_keys or count > config.max_keys:
            return False
        if config.require_simultaneous and group.max_simultaneous < count:
            # Not every key was down together, so this was typing that happened to
            # overlap rather than a deliberate press.
            return False
        if group.held and t_ms - group.t0 >= config.chord_max_ms:
            # Still held when the time ran out: fingers resting on the keyboard, or a
            # gesture too slow to have been meant as one.
            return False
        if count == 2:
            release = group.first_release_ms()
            overlap = (release if release is not None else t_ms) - group.last_down_ms
            if overlap < config.two_key_overlap_ms:
                return False
        return True

    def _flush_as_letters(self, group: Group, *, synthetic_up: bool) -> list[Action]:
        if self.config.trigger is Trigger.OPTIMISTIC:
            return []  # already typed; nothing left to send
        if not group.events:
            return []
        if not synthetic_up:
            # The user is still holding at least one key, so let the real releases through
            # and key repeat keeps working.
            for key in group.held:
                self.pending_release.discard(key)
        return [TypeLetters(events=tuple(group.events), synthetic_up=synthetic_up)]

    # -- hold-to-chord modifier ----------------------------------------------

    def set_chord_modifier(self, held: bool, t_ms: float = 0.0) -> list[Action]:
        """Report the hold-to-chord key going down or up."""
        self.chord_modifier_down = held
        if not held and self.group is not None:
            return self._resolve(t_ms)
        return []
