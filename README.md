# stenographer

Type a word by pressing several of its letters at once, like a stenographer, and let a
model work out which word you meant from those letters and the sentence so far.

Press `t`, `h` and `e` together and you get "the". Press `b`, `c` and `s` together and
you get "because". Press one letter at a time and nothing changes: single presses are
passed straight through, so ordinary typing always works.

## Does it actually save time?

Not automatically, and that is the whole question. Fewer keystrokes is not the same as
less time, because a wrong first guess costs you the time to notice it and the time to
repair it. Published work on word suggestion has found systems that cut keystrokes while
making able-bodied typists slower overall.

So the arithmetic came first. `scripts/evaluate.py` measures how ambiguous chords really
are and how well they can be resolved, and `model/speed_model.py` turns that into a
verdict. With a 200 ms keystroke (about 60 words per minute) and a 600 ms chord, a
six-letter word needs roughly 68% top-1 accuracy before chording wins at all.

Two measurements decide it, and neither is a guess:

| | how to get it | status |
|---|---|---|
| How accurately a chord can be resolved | `scripts/evaluate.py`, `scripts/score_model.py` | measured, below |
| How fast your hands form a chord | `scripts/chord_drill.py` | needs you at the keyboard |

### What the numbers say so far

Over 20,000 prediction sites from held-out WikiText-103, with chords drawn from a mixture
of briefing habits:

| ranking | top-1 | top-3 | top-5 |
|---|---|---|---|
| frequency only, no context | 43.5% | 58.0% | 62.2% |
| trained model, after 256k examples | 65.4% | 74.9% | 78.0% |

Both were scored on identical trials, so the gap is what context bought and nothing else.
That model had seen 4% of a full training run; validation accuracy was still climbing
steeply when it was measured (60.3% at 1,000 steps, 67.4% at 2,000).

Frequency alone splits sharply by how common the word is: 80% top-1 for the hundred most
frequent words, 4.5% for those ranked one to ten thousand. Closing that gap is the
model's entire job, and after 4% of a training run it had taken the middle band from 4.5%
to 28.6% and two-letter chords to 78.7%.

At that accuracy the speed model already says chording wins for words of five letters or
more, by 5.6% at five letters and 22.9% at seven, and loses on four-letter words. Which
is the expected shape: short words are not worth chording.

Chords are genuinely ambiguous. Over the 50,000-word vocabulary, the median 3-letter
chord matches around 1,900 words and `{e,h,t}` alone matches "the", "they", "their",
"there", "them" and "other". That is why ranking scores every candidate in one pass
rather than scoring them one at a time, and why context matters more than chord length.

## Getting started

```bash
python3.14 -m venv .venv
.venv/bin/pip install -e ".[ml,mac,dev]"
.venv/bin/python scripts/build_vocab.py       # 50k words with letter masks
.venv/bin/python scripts/prepare_corpus.py    # WikiText-103 to local text files
.venv/bin/python scripts/evaluate.py          # ambiguity, baseline accuracy, speed verdict
```

Then measure your own hands, which needs no permissions:

```bash
.venv/bin/python scripts/chord_drill.py
```

### Training a model

```bash
.venv/bin/python scripts/train.py --steps 25000 --batch-size 256 --out runs/mine
.venv/bin/python scripts/score_model.py --checkpoint runs/mine/checkpoint.pt
```

On an M5 Pro this runs at about 370 examples a second, so 25,000 steps at batch 256 takes
roughly five hours. `scripts/score_model.py` scores the checkpoint against the frequency
baseline on identical trials, so the difference is what context bought and nothing else.

### Running it for real

Two processes. The daemon holds the model; the tap watches the keyboard.

```bash
.venv/bin/python scripts/check_permissions.py             # do this first
.venv/bin/python scripts/stenod.py --checkpoint runs/mine/checkpoint.pt
.venv/bin/python scripts/steno.py --dry-run               # watch, change nothing
.venv/bin/python scripts/steno.py                         # for real
```

**Start with `--dry-run`.** It watches without touching anything and prints what it would
have done. It is the only safe way to find out whether the timing thresholds suit your
hands before letting it rewrite what you type.

## Permissions

This needs Accessibility permission, and without it the failure is silent.
`CGEventTapCreate` returns a valid tap, `CGEventTapIsEnabled` reports it as enabled, and
no keystroke is ever delivered. That is why `scripts/check_permissions.py` posts a
synthetic keystroke and waits to see it arrive, rather than trusting either call.

The permission belongs to whichever process macOS holds responsible. Run from a terminal
and that is the terminal application, not Python, and its child processes inherit the
grant. Add it under System Settings, Privacy & Security, Accessibility.

Do not revoke the permission while the tap is running. On current macOS that hangs
keyboard input with no notification, which is why a health check stands the tap down as
soon as it stops receiving events.

While a password field has the keyboard, Secure Input stops events reaching any tap at
all. Nothing is captured and nothing is logged.

## How it works

```
keyboard ─▶ steno.py                                    stenod.py
             tap ─▶ detector ─▶ runtime ─▶ emitter        model + context
                                   │                            ▲
                                   └── chord ─── socket ────────┘
                                       50 ms deadline, then the local index
```

**Two processes, deliberately.** A process holding an active event tap freezes keyboard
input for every application on the machine if its callback stalls, and the system's
recovery from that is unreliable. Torch holds the interpreter lock through the Python side
of a forward pass, and an Accessibility read can block for seconds. Neither belongs
anywhere near the tap.

**The tap always has an answer.** If the daemon is slow, crashed, or was never started,
the tap falls back to the most frequent word containing the chord letters, found by
intersecting 26 bitsets in about 40 microseconds. Stopping the daemon makes chording less
accurate; it never stops typing. `kill -STOP` the daemon mid-sentence and keystrokes keep
flowing.

**Telling a chord from fast typing** is the hard part. Fast typists overlap a large share
of consecutive keystrokes, so "two keys were down together" means very little on its own.
The detector weighs three things: how closely the presses arrived (50 ms to join a group),
how many keys were down at one instant, and how long a pair overlapped before the first
release (60 ms). Keys still held after 600 ms are fingers resting on the keyboard, not a
chord.

Two alternatives are one config flag away, because which one suits real hands is a
question for the dry run rather than for a design document. `--trigger optimistic` types
letters immediately and takes them back when a chord forms, costing no latency but
flickering. `--trigger hold_to_chord` only forms chords while a modifier is held, which
cannot produce a false chord at all.

**Around the word.** A committed word gets a space after it. Punctuation typed next pulls
that space back so "there" plus a comma gives "there, ". A second space is swallowed. Tab
steps through the alternatives; Backspace removes the whole word. Casing comes from the
vocabulary first, then Shift and Caps Lock, then sentence position, and stays lowercase
when the evidence is thin, because capitalising an ordinary word mid-sentence is the most
visible way this can look broken.

## Layout

| path | what it holds |
|---|---|
| `model/` | vocabulary, candidate generation, synthetic chords, corpus, training, evaluation |
| `stenographer/keyboard/` | detector, event tap, injection, layout, composer |
| `stenographer/brain/` | daemon, socket client, predictors |
| `stenographer/text/` | casing rules |
| `stenographer/runtime.py` | the keystroke path, from key event to injected text |
| `stenographer/protocol.py` | the messages between the two processes |
| `scripts/` | everything runnable |
| `tests/` | 428 tests, none of which need macOS |

## Tests

```bash
.venv/bin/python -m pytest
```

The parts most likely to be subtly wrong are the parts with no macOS dependency, which is
deliberate. The detector is a pure state machine driven by written-out timings, so the
difference between a chord and a rollover is pinned down in milliseconds. The composer's
tests apply its operations to a simulated text field and assert on what a user would see,
because every repair is a backspace count and an off-by-one there silently eats their
text. The integration tests drive the real runtime with a fake emitter.

## Design notes

Some things turned out differently from the plan, and the reasons are worth keeping.

**Substitution noise in training is off.** Candidate generation keeps a word only when it
contains every chord letter, so a chord carrying a letter the target lacks excludes that
target outright, and the model's output is masked to the candidates. Training on such an
example teaches it to back wrong words and can never teach it the right one. A slipped
finger is handled by the cycle and undo keys instead.

**Typability resampling is off.** Rerolling an awkward chord to a random subset discards
the prefix or skeleton structure that made it predictable, and scored less than half the
accuracy of the policies it replaced. Typability is reported by the harness instead. Note
that around half of 4-letter prefix chords put two letters under one finger, which is a
real cost the drill will show.

**The corpus is materialised to local files first.** Reading the training split straight
from the Hugging Face cache fetches it lazily in chunks; a run once spent 46 minutes on
300 steps with the process idle at 5% CPU, waiting on the network.

**bf16 doubles training throughput** on this machine, 445 against 229 examples a second,
and the 50,000-word output head is nearly free: with a 1,000-word head it reaches 457. The
encoder is the cost, so shrinking the vocabulary would buy nothing.

**Text is emitted by overriding an event's Unicode string,** not by pressing the keys that
would produce it. An event's string is computed when the event is created and setting the
Shift flag afterwards does not recompute it, so a shifted-character table cannot be built
that way at all.

## Where this stands

Working and verified: the offline harness, the vocabulary and candidate index, synthetic
chords, the detector, the composer and casing, the runtime, the two processes talking over
their socket with the fallback path under a killed daemon, and the permission check.

Not yet verified: anything that needs Accessibility permission, which is not granted on
this machine. The tap creates and reports itself enabled but receives nothing, so the live
capture and injection path is written and unit-tested but has never run against real
keystrokes. Grant the permission, run `scripts/check_permissions.py` until it says ready,
then `scripts/steno.py --dry-run`.

Not built: reading context from the focused application through the Accessibility API, the
per-application allow and deny rules, the menubar indicator, and the launch agents. The
daemon tracks context from what the tap reports typing, which is enough for a field you
type into from start to finish but not for one you click into halfway.
