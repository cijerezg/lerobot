"""Raw capture of the 102HD leader bus (pi07_wiki/leader_bus_investigation.md §5.1-5.4).

Sends the SDK's byte-identical sync-monitor request (or single monitor queries), records every
byte that comes back with arrival times, and judges each burst against the wire instead of
through the SDK collector: which frames are present, at which offsets, whether each checksum
passes, how many bytes are left over, whether anything precedes the first header (echo), and
when each frame landed. Raw bytes of every trial go to a JSON under outputs/ (run from the workspace root).

    .venv/bin/python lerobot/scripts/leader_bus_capture.py suite            # everything, ~20 s
    .venv/bin/python lerobot/scripts/leader_bus_capture.py burst --ids 6,5,4,3,2,1,0 --trials 20
    .venv/bin/python lerobot/scripts/leader_bus_capture.py single --trials 5
    .venv/bin/python lerobot/scripts/leader_bus_capture.py burst --trials 200 --spacing 0.033 --window 0.025

Arm powered, torque off, nothing else on the port. Read-only on the bus.
"""

import argparse
import json
import statistics
import struct
import time
from collections import Counter
from pathlib import Path

import serial
from fashionstar_uart_sdk.uservo import Packet, UartServoManager as M

PORT = "/dev/ttyUSB0"
BAUD = 1_000_000
FRAME = 21  # header 2 + code 1 + size 1 + params 16 (<BHHHHBih) + checksum 1
DEFAULT_IDS = [0, 1, 2, 3, 4, 5, 6]


def sync_request(ids):
    param = struct.pack("<BBB", M.CODE_QUERY_SERVO_MONITOR, 1, len(ids)) + bytes(ids)
    return Packet.pack(M.CODE_SYNC_COMMAND, param)


def single_request(sid):
    return Packet.pack(M.CODE_QUERY_SERVO_MONITOR, struct.pack("<B", sid))


def capture(u, req, window):
    """Send req, poll the port for `window` seconds; return (bytes, [(t_ms, nbytes), ...])."""
    u.reset_input_buffer()
    t0 = time.perf_counter()
    u.write(req)
    buf = bytearray()
    chunks = []
    while True:
        n = u.in_waiting
        if n:
            data = u.read(n)
            chunks.append(((time.perf_counter() - t0) * 1000.0, len(data)))
            buf += data
        elif time.perf_counter() - t0 > window:
            break
    return bytes(buf), chunks


def parse_frames(buf):
    """Every response header in buf with its verdict. A valid frame is skipped whole; anything
    else advances one byte so a damaged frame still exposes the next good header."""
    frames = []
    i = 0
    while i < len(buf) - 1:
        if buf[i] == 0x05 and buf[i + 1] == 0x1C:
            size = buf[i + 3] if i + 3 < len(buf) else None
            end = None if size is None else i + 5 + size
            if end is not None and end <= len(buf):
                fb = buf[i:end]
                ok = Packet.verify(fb, pkt_type=Packet.PKT_TYPE_RESPONSE)
                sid = buf[i + 4] if size else None
                frames.append({"off": i, "end": end, "code": buf[i + 2], "size": size, "id": sid, "ok": ok})
                if ok:
                    i = end
                    continue
            else:
                frames.append({"off": i, "end": len(buf), "code": None, "size": size, "id": None, "ok": False, "truncated": True})
        i += 1
    return frames


def arrival_ms(chunks, offset):
    """Arrival time of the byte at `offset`, from the chunk that carried it."""
    seen = 0
    for t, n in chunks:
        seen += n
        if offset < seen:
            return t
    return None


def judge(buf, chunks, ids):
    frames = parse_frames(buf)
    good = [f for f in frames if f["ok"]]
    bad = [f for f in frames if not f["ok"]]
    got = [f["id"] for f in good]
    first_hdr = frames[0]["off"] if frames else None
    contiguous = all(f["off"] == first_hdr + k * FRAME for k, f in enumerate(good)) if good else False
    last_end = good[-1]["end"] if good else 0
    return {
        "n": len(buf),
        "first_ms": chunks[0][0] if chunks else None,
        "last_ms": chunks[-1][0] if chunks else None,
        "n_chunks": len(chunks),
        "ids": got,
        "missing": [i for i in ids if i not in got],
        "extra": [i for i in got if i not in ids],
        "dup": [i for i, c in Counter(got).items() if c > 1],
        "bad": [(f["off"], f["id"], f.get("size")) for f in bad],
        "pre": first_hdr if first_hdr is not None else len(buf),
        "leftover": len(buf) - last_end,
        "contiguous": contiguous,
        "frame_ms": [arrival_ms(chunks, f["end"] - 1) for f in good],
        "expected": len(ids) * FRAME,
    }


def fmt_trial(k, j):
    s = (f"trial {k:3d}: {j['n']:3d} B/{j['expected']}  first {j['first_ms'] or 0:5.2f} last {j['last_ms'] or 0:5.2f} ms"
         f"  chunks {j['n_chunks']:2d}  ids {j['ids']}")
    if j["missing"]:
        s += f"  MISSING {j['missing']}"
    if j["bad"]:
        s += f"  BAD(off,id,size) {j['bad']}"
    if j["pre"]:
        s += f"  PRE {j['pre']}B"
    if j["leftover"]:
        s += f"  LEFTOVER {j['leftover']}B"
    if j["extra"] or j["dup"]:
        s += f"  extra {j['extra']} dup {j['dup']}"
    if not j["contiguous"]:
        s += "  NONCONTIG"
    return s


def summarize(label, judged, ids, head=0):
    n = len(judged)
    complete = [j for j in judged if not j["missing"] and not j["bad"] and not j["leftover"] and not j["pre"]]
    print(f"\n== {label}: {len(complete)}/{n} bursts clean (all {len(ids)} frames valid, contiguous, no extra bytes)")
    miss = Counter(i for j in judged for i in j["missing"])
    if miss:
        print(f"   missing per id: {dict(sorted(miss.items()))}")
        print(f"   missing sets: {Counter(tuple(j['missing']) for j in judged if j['missing']).most_common(8)}")
    print(f"   byte counts: {sorted(Counter(j['n'] for j in judged).items())}")
    badpos = Counter(off // FRAME for j in judged for off, _, _ in j["bad"])
    if badpos:
        print(f"   damaged frames by slot (offset//21): {dict(sorted(badpos.items()))}")
        print(f"   damaged (off,id,size) examples: {[b for j in judged for b in j['bad']][:12]}")
    print(f"   bursts with bytes before first header: {sum(1 for j in judged if j['pre'])}, "
          f"with leftover bytes: {sum(1 for j in judged if j['leftover'])}, "
          f"non-contiguous: {sum(1 for j in judged if not j['contiguous'])}")
    firsts = [j["first_ms"] for j in judged if j["first_ms"] is not None]
    lasts = [j["last_ms"] for j in judged if j["last_ms"] is not None]
    if firsts:
        print(f"   first byte ms: median {statistics.median(firsts):.2f} max {max(firsts):.2f};"
              f" last byte ms: median {statistics.median(lasts):.2f} max {max(lasts):.2f}")
    if complete:
        per_slot = list(zip(*[j["frame_ms"] for j in complete]))
        print("   frame arrival ms (median over clean bursts): "
              + " ".join(f"{ids[k]}:{statistics.median(v):.2f}" for k, v in enumerate(per_slot)))
    if head and n > head:
        a = sum(1 for j in judged[:head] if not (j["missing"] or j["bad"]))
        b = sum(1 for j in judged[head:] if not (j["missing"] or j["bad"]))
        print(f"   warm-up check, clean bursts: first {head}: {a}/{head}; rest: {b}/{n - head}")


def run_burst(u, ids, trials, spacing, window, verbose, hexdump):
    req = sync_request(ids)
    print(f"request ({len(req)} B): {req.hex(' ')}")
    raw, judged = [], []
    for k in range(trials):
        t_iter = time.perf_counter()
        buf, chunks = capture(u, req, window)
        j = judge(buf, chunks, ids)
        raw.append({"buf": buf.hex(), "chunks": chunks})
        judged.append(j)
        damaged = j["missing"] or j["bad"] or j["leftover"] or j["pre"]
        if verbose or damaged:
            print(fmt_trial(k, j))
        if hexdump and (hexdump == "all" or damaged):
            print("    " + buf.hex(" "))
        rest = spacing - (time.perf_counter() - t_iter)
        if rest > 0:
            time.sleep(rest)
    return raw, judged


def run_single(u, ids, trials, spacing, window, verbose, hexdump):
    raw, judged = {}, {}
    for sid in ids:
        req = single_request(sid)
        raw[sid], judged[sid] = [], []
        for k in range(trials):
            buf, chunks = capture(u, req, window)
            j = judge(buf, chunks, [sid])
            raw[sid].append({"buf": buf.hex(), "chunks": chunks})
            judged[sid].append(j)
            if verbose or j["missing"] or j["bad"]:
                print(f"id {sid} " + fmt_trial(k, j))
            if hexdump == "all":
                print("    " + buf.hex(" "))
            time.sleep(spacing)
        firsts = [j["first_ms"] for j in judged[sid] if j["first_ms"] is not None]
        lasts = [j["last_ms"] for j in judged[sid] if j["last_ms"] is not None]
        ok = sum(1 for j in judged[sid] if not j["missing"] and not j["bad"])
        print(f"single id {sid}: {ok}/{trials} valid; first byte median {statistics.median(firsts) if firsts else float('nan'):.2f} ms"
              f" max {max(firsts) if firsts else float('nan'):.2f}; last byte median {statistics.median(lasts) if lasts else float('nan'):.2f} ms;"
              f" bytes {sorted(Counter(j['n'] for j in judged[sid]).items())}")
    return raw, judged


WILD = {5, 6, 20}  # mV lo/hi and the checksum differ between bursts even at rest


def align_deletion(frag, template):
    """Which template indices are absent from frag, assuming frag = template with one contiguous
    run deleted: longest matching prefix + longest matching suffix, value bytes in WILD ignored.
    None if the fragment is not explained by a single deletion."""
    j = 0
    while j < len(frag) and j < len(template) and (j in WILD or frag[j] == template[j]):
        j += 1
    k = 0
    while k < len(frag) - j and ((len(template) - 1 - k) in WILD or frag[-1 - k] == template[-1 - k]):
        k += 1
    if j + k >= len(frag):
        return (j, len(template) - k)  # deleted [j, len-k)
    return None


def analyze(path, verbose):
    data = json.loads(path.read_text())
    print(f"capture {path.name} started {data['started']}")
    for run in data["runs"]:
        if run["mode"] != "burst":
            continue
        ids = run["ids"]
        label = run.get("label", f"burst ids={ids}")
        trials = [(bytes.fromhex(r["buf"]), r["chunks"]) for r in run["raw"]]
        judged = [judge(b, c, ids) for b, c in trials]
        summarize(label, judged, ids, head=20 if len(judged) >= 50 else 0)
        # slot vs id: for each burst, which slots (position in the request) lost their frame
        slot_of = {sid: k for k, sid in enumerate(ids)}
        miss_slot = Counter(slot_of[i] for j in judged for i in j["missing"])
        print(f"   missing by slot (0=first in request): {dict(sorted(miss_slot.items()))}   ids at slots: {ids}")
        # deletion alignment against the same servo's clean frame from the nearest clean burst
        clean_frames = {}
        for b, _ in trials:
            for f in parse_frames(b):
                if f["ok"]:
                    clean_frames.setdefault(f["id"], b[f["off"]:f["end"]])
        del_hist = Counter()
        frag_len = Counter()
        unexplained = []
        starts = Counter()
        tails = 0
        for t, (b, c) in enumerate(trials):
            frames = parse_frames(b)
            good = [f for f in frames if f["ok"]]
            # fragments = bytes between consecutive good frames (and before/after)
            cuts = [0] + [x for f in good for x in (f["off"], f["end"])] + [len(b)]
            for a, e in zip(cuts[0::2], cuts[1::2]):
                if e <= a:
                    continue
                frag = b[a:e]
                frag_len[len(frag)] += 1
                # which id is this fragment? the missing id whose slot sits between the neighbours
                prev_ids = [f["id"] for f in good if f["end"] <= a]
                exp_slot = slot_of[prev_ids[-1]] + 1 if prev_ids else 0
                sid = ids[exp_slot] if exp_slot < len(ids) else None
                tpl = clean_frames.get(sid)
                al = align_deletion(frag, tpl) if tpl else None
                tag = f"slot {exp_slot} id {sid}"
                if al:
                    del_hist[(exp_slot, sid, al)] += 1
                    starts[exp_slot * FRAME + al[0]] += 1
                    if exp_slot * FRAME + al[1] == len(ids) * FRAME and e == len(b):
                        tails += 1
                    if verbose:
                        print(f"   trial {t:3d} {tag}: {len(frag):2d} B kept, deleted frame bytes [{al[0]},{al[1]}) at ~{arrival_ms(c, a):.2f} ms")
                else:
                    unexplained.append((t, tag, len(frag), frag.hex(" ")))
        print(f"   fragment lengths (bytes kept of a damaged 21-B frame; 0 = frame fully absent): {sorted(frag_len.items())}")
        print(f"   deletion START offsets in the stream: {sorted(starts.items())}  (mod 32: {sorted(Counter(o % 32 for o, n in starts.items() for _ in range(n)).items())}); lost to end of burst: {tails}")
        print("   deletions (slot, id, [from,to) frame byte index) -> count:")
        for (slot, sid, (a, e)), n in sorted(del_hist.items(), key=lambda kv: -kv[1])[:12]:
            print(f"      slot {slot} id {sid}: bytes [{a:2d},{e:2d}) x{n}")
        if unexplained:
            print(f"   not a single contiguous deletion ({len(unexplained)}):")
            for t, tag, n, hx in unexplained[:8]:
                print(f"      trial {t} {tag} {n} B: {hx}")
        sizes = Counter(n for _, c in trials for _, n in c)
        print(f"   read chunk sizes (bytes per poll): {sorted(sizes.items())[:20]}")


def _spin():
    while True:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["burst", "single", "suite", "analyze"])
    ap.add_argument("--json", default=None, help="analyze: saved capture to reload")
    ap.add_argument("--ids", default=",".join(map(str, DEFAULT_IDS)))
    ap.add_argument("--trials", type=int, default=8)
    ap.add_argument("--spacing", type=float, default=0.12, help="seconds between sends")
    ap.add_argument("--window", type=float, default=0.05, help="seconds to keep collecting after the send")
    ap.add_argument("--port", default=PORT)
    ap.add_argument("-v", "--verbose", action="store_true", help="print every trial, not only damaged ones")
    ap.add_argument("--hex", choices=["damaged", "all"], default="damaged")
    ap.add_argument("--out", default=None)
    ap.add_argument("--busy", type=int, default=0, help="spawn N spinning processes so cores never enter deep C-states")
    args = ap.parse_args()
    ids = [int(x) for x in args.ids.split(",")]
    if args.mode == "analyze":
        analyze(Path(args.json), args.verbose)
        return

    burners = []
    if args.busy:
        import multiprocessing
        for _ in range(args.busy):
            pr = multiprocessing.Process(target=_spin, daemon=True)
            pr.start()
            burners.append(pr)
        time.sleep(0.5)
        print(f"{args.busy} busy processes spinning")
    u = serial.Serial(args.port, BAUD, timeout=0)
    t_open = time.perf_counter()
    out = {"port": args.port, "baud": BAUD, "busy": args.busy, "started": time.strftime("%Y-%m-%d %H:%M:%S"), "runs": []}
    try:
        if args.mode == "burst":
            raw, judged = run_burst(u, ids, args.trials, args.spacing, args.window, args.verbose, args.hex)
            summarize(f"burst ids={ids} x{args.trials} @{args.spacing*1000:.0f} ms", judged, ids, head=min(20, args.trials // 4))
            out["runs"].append({"mode": "burst", "ids": ids, "spacing": args.spacing, "window": args.window, "raw": raw})
        elif args.mode == "single":
            raw, judged = run_single(u, ids, args.trials, args.spacing, args.window, args.verbose, args.hex)
            out["runs"].append({"mode": "single", "ids": ids, "spacing": args.spacing, "window": args.window, "raw": raw})
        else:
            plan = [
                ("A open-port + baseline 0..6", DEFAULT_IDS, 50, 0.12, 0.05),
                ("B burst size 4 (0..3)", [0, 1, 2, 3], 20, 0.12, 0.05),
                ("C reversed order 6..0", list(reversed(DEFAULT_IDS)), 20, 0.12, 0.05),
                ("D gripper first 6,0..5", [6, 0, 1, 2, 3, 4, 5], 20, 0.12, 0.05),
                ("E runtime cadence 0..6 @33 ms", DEFAULT_IDS, 200, 0.033, 0.025),
            ]
            for label, pids, trials, spacing, window in plan:
                print(f"\n### {label}  (t+{time.perf_counter() - t_open:.1f} s after open)")
                raw, judged = run_burst(u, pids, trials, spacing, window, args.verbose, args.hex)
                summarize(label, judged, pids, head=20 if trials >= 50 else 0)
                out["runs"].append({"mode": "burst", "label": label, "ids": pids, "spacing": spacing, "window": window, "raw": raw})
            print("\n### F single monitor queries x5 per id")
            raw, judged = run_single(u, DEFAULT_IDS, 5, 0.05, 0.05, args.verbose, args.hex)
            out["runs"].append({"mode": "single", "ids": DEFAULT_IDS, "spacing": 0.05, "window": 0.05, "raw": raw})
    finally:
        u.close()
        for pr in burners:
            pr.terminate()
        path = Path(args.out) if args.out else Path("outputs") / f"leader_bus_capture_{time.strftime('%H%M%S')}.json"
        path.write_text(json.dumps(out))
        print(f"\nraw capture saved: {path}")


if __name__ == "__main__":
    main()
