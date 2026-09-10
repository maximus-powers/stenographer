# stenographer

Type a word by pressing its opening letters at once, like a stenographer, and let a
model work out which word you meant from those letters and the sentence so far.

**A chord is the word's first letters, in any order.** Press `t`, `h` and `e` together
and you get "the ", space included. Press `b`, `e` and `c` and you get "because ", the
first three letters of it. Four keys means the first four: `{b,e,c,a}` is still because,
and `{t,o,m,r}` is tomorrow. Wrong word? Press the same chord again and it steps to the
next candidate, so `{e,h,t}` twice gives "they" and a third time "their". Press one
letter at a time and nothing changes: single presses are passed straight through, so
ordinary typing always works.

There is nothing to memorise, and nothing to learn per word: the chord for a word is the
start of the word. A doubled letter costs no key, since a hand cannot press it twice at
once, so the first four of "little" are l, i, t, e.

You have 300 ms to form a chord, measured from the first key going down to the last one
coming up, which leaves about 220 ms for your fingers to arrive. There is no need to hit
the keys at the same instant, only to have them all down together at some point before
you lift.

## Does it actually save time?

Not automatically, and that is the whole question. Fewer keystrokes is not the same as
less time, because a wrong first guess costs you the time to notice it and the time to
repair it. Published work on word suggestion has found systems that cut keystrokes while
making able-bodied typists slower overall.

So the arithmetic came first. `scripts/evaluate.py` measures how ambiguous chords really
are and how well they can be resolved, and `model/speed_model.py` turns that into a
verdict. With a 200 ms keystroke (about 60 words per minute) and a 600 ms chord, a
six-letter word needs roughly 68% top-1 accuracy before chording wins at all.

That 600 ms is an assumption, and it is now in open conflict with the 300 ms the detector
allows: a gesture the speed model treats as normal would not be recognised as a chord at
all. One of the two numbers is wrong, and `scripts/chord_drill.py` is what settles it,
because it measures how long a hand actually takes. Until it has been run, treat the
window as the claim under test.

Two measurements decide it, and neither is a guess:

| | how to get it | status |
|---|---|---|
| How accurately a chord can be resolved | `scripts/evaluate.py`, `scripts/score_model.py` | measured, below |
| How fast your hands form a chord | `scripts/chord_drill.py` | needs you at the keyboard |

### What the numbers say so far

Over 20,000 prediction sites from held-out WikiText-103, chording each word by its
opening letters:

| ranking | top-1 | top-3 | top-5 |
|---|---|---|---|
| frequency only, no context | 63.5% | 78.9% | 84.9% |
| trained model | being retrained | | |

The frequency number is what the system scores with no model at all, and most of it is
the candidate rule rather than the ranker: matching any word that merely contained the
chord letters scored 43.5% on the same trials. Restricting a chord to a word's opening
letters is worth twenty points before a model is consulted.

An earlier model reached 81.7% top-1 after 25,000 steps, but it was trained when 65% of
its examples were briefs this no longer accepts, and scored against a candidate set about
twenty times larger. Neither number carries over, so the row above waits for the retrain.
Its shape is still worth knowing: validation accuracy was climbing when that run ended
(81.0% at 21,000 steps, 81.7% at 25,000), so the ceiling was not reached.

Frequency alone splits sharply by how common the word is, and closing that gap is the
model's entire job.

Chords are still ambiguous, just far less so. Over the 50,000-word vocabulary the median
3-key chord opens 9 words and the median 4-key chord opens 5, against roughly 1,900 and
600 when a chord meant any word containing those letters. `{e,h,t}` gives "the", "they",
"their", "there", "them", "then", "these" - but no longer "other", which contains all
three and opens with none of them. That is why ranking scores every candidate in one pass
rather than scoring them one at a time, and why context still matters more than chord
length: the words a chord opens are exactly the words that look alike at the start.

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

```bash
./steno --dry-run    # watch and print what it would have typed, change nothing
./steno              # for real, in every application
./steno --stop       # stop a daemon left running in the background
```

**Start with `--dry-run`.** It watches without touching anything and prints what it would
have done. It is the only safe way to find out whether the timing thresholds suit your
hands before letting it rewrite what you type.

Once it is running it is running everywhere: an event tap sees the keyboard for the whole
login session, not one application. Ctrl-C stops it.

`./steno` checks Accessibility permission before it loads anything, starts the prediction
daemon if one is not already up, waits for the model, and runs the tap in front. Flags it
does not recognise go to the tap, so `./steno --dry-run --chord-max 700` works; `--cpu`
and `--no-model` go to the daemon. The daemon's own output goes to `runs/stenod.log`.

There are two processes because a tap that stalls freezes the keyboard for every
application on the machine, so the model may not share with it. To run them by hand:

```bash
.venv/bin/python scripts/check_permissions.py             # do this first
.venv/bin/python scripts/stenod.py --checkpoint runs/mine/checkpoint.pt
.venv/bin/python scripts/steno.py --dry-run
```

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

That deadline is not theoretical. Asked six chords while a training run had the GPU, the
daemon answered four inside 50 ms at about 10 ms each and missed two, and those two fell
back to the local index instead of stalling. A late reply is matched against the chord it
was asked about and discarded if it belongs to a request already given up on, so a slow
answer can never be attributed to the wrong chord.

**Telling a chord from fast typing** is the hard part, and the answer is not to measure
speed. A chord gets half a second, and what marks it out instead is that every one of its
keys was down at the same moment. A typist releases each key about as the next goes down,
so ordinary typing never gets three keys down together however fast it goes, while a
deliberate press can be made as slowly as the hand likes. Keys still down when the window
runs out are fingers resting on the keyboard, not a chord.

The window is not what keeps ordinary typing out, which is why it can be set to what a
hand needs rather than to the widest value that stays safe. It has to cover the whole
gesture including the release, so half a second means fingers arriving over about 420 ms.

Two keys are the weak case, because a pair overlapping is exactly what rollover looks
like, so a pair also has to be held together for 60 ms to count. All three numbers are
flags: `--chord-max`, `--lone-hold`, `--overlap`.

Waiting for your fingers to lift costs ordinary typing nothing, which is easy to get
wrong. A single keystroke is decided the moment it is released, not when the window
expires, so the letter appears as the finger comes up. What it does affect is holding a
letter key down to repeat it, where the first character waits out `--lone-hold`.

Two alternatives are one config flag away, because which one suits real hands is a
question for the dry run rather than for a design document. `--trigger optimistic` types
letters immediately and takes them back when a chord forms, costing no latency but
flickering. `--trigger hold_to_chord` only forms chords while a modifier is held, which
cannot produce a false chord at all.

**Around the word.** A committed word gets a space after it, so chords run together into
a sentence without touching the space bar. Punctuation typed next pulls that space back,
so "there" plus a comma gives "there, ". A second space is swallowed.

**Fixing a wrong guess.** Press the same chord again and it steps to the next candidate,
which needs no new key and leaves your fingers where they are. Tab does the same thing,
Shift-Tab goes back, and Backspace removes the word entirely. A repeat only counts as a
correction within a second and a half, so a word can still be typed twice: chord, space,
chord. Casing comes from the
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
| `steno` | one command: permission check, daemon, tap |
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
