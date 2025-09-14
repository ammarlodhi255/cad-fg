import os, re, glob
import pandas as pd

def _read_lines(path):
    out = []
    if os.path.isfile(path):
        with open(path) as fh:
            for line in fh:
                t = line.strip().lower()
                if t:
                    out.append(t)
    return out

def _parse_video_keys(lines):
    """
    Accepts: 'p12_v3', 'p12/v3', 'p12 v3', 'p12-v3', or just 'v3'
    Returns (pairs, vonly): pairs={(p,v)}, vonly={v}
    """
    pairs = set()
    vonly = set()
    for t in lines:
        m = re.search(r"p\s*(\d+)\s*[_/\-\s]\s*v\s*(\d+)", t)
        if m:
            pairs.add((int(m.group(1)), int(m.group(2))))
            continue
        m2 = re.match(r"^v\s*(\d+)$", t)
        if m2:
            vonly.add(int(m2.group(1)))
            continue
        m3 = re.findall(r"\d+", t)
        if len(m3) == 2:
            pairs.add((int(m3[0]), int(m3[1])))
        elif len(m3) == 1 and t.startswith("v"):
            vonly.add(int(m3[0]))
    return pairs, vonly

def _read_selected_frames(patient_dir, video_dir, p_id, v_id):
    """p{p}_v{v}_selectedFrames.txt can be in video_dir OR patient_dir."""
    f_video   = os.path.join(video_dir,   f"p{p_id}_v{v_id}_selectedFrames.txt")
    f_patient = os.path.join(patient_dir, f"p{p_id}_v{v_id}_selectedFrames.txt")
    path = f_video if os.path.isfile(f_video) else f_patient if os.path.isfile(f_patient) else None
    ids = []
    if path:
        with open(path) as fh:
            for line in fh:
                nums = re.findall(r"\d+", line)
                ids.extend(int(n) for n in nums)
    return ids

def _index_input_frames(inp_dir):
    """
    Build rich indices for mapping selected frame numbers -> filenames.
    Uses LAST integer in filenames (handles p1_v1_00023.png, frame00023.png, 1.png, etc).
    Also builds zero-padded suffix and exact-stem maps; supports png/jpg/jpeg.
    """
    num_map = {}
    pad_map = {}
    ext_map = {}
    try:
        names = os.listdir(inp_dir)
    except FileNotFoundError:
        return {"num_map": num_map, "pad_map": pad_map, "ext_map": ext_map, "names": []}

    for fname in names:
        stem, ext = os.path.splitext(fname)
        ext_map[stem.lower()] = fname
        nums = re.findall(r"(\d+)", fname)
        if nums:
            last_idx = int(nums[-1])
            num_map.setdefault(last_idx, fname)
            pad_map.setdefault(f"_{int(nums[-1]):05d}", fname)
    return {"num_map": num_map, "pad_map": pad_map, "ext_map": ext_map, "names": names}

def build_index(root="data/CADICA"):
    base = os.path.join(root, "selectedVideos")
    if not os.path.isdir(base):
        raise FileNotFoundError(f"Expected directory not found: {base}")

    # GLOBAL lesion/nonlesion lists (if any)
    g_les_lines = _read_lines(os.path.join(base, "lesionVideos.txt"))
    g_non_lines = _read_lines(os.path.join(base, "nonlesionVideos.txt"))
    g_pairs_les, g_vonly_les = _parse_video_keys(g_les_lines)
    g_pairs_non, g_vonly_non = _parse_video_keys(g_non_lines)

    patients = sorted(glob.glob(os.path.join(base, "p*")))
    if not patients:
        raise RuntimeError(f"No patient folders like p*/ found under {base}.")

    rows = []
    unlabeled = []
    no_input = []
    no_kf = []
    missing_map = []

    for patient_dir in patients:
        mp = re.search(r"p(\d+)$", os.path.basename(patient_dir))
        if not mp:
            continue
        p_id = int(mp.group(1))

        # PER-PATIENT lists (preferred if present)
        p_les_lines = _read_lines(os.path.join(patient_dir, "lesionVideos.txt"))
        p_non_lines = _read_lines(os.path.join(patient_dir, "nonlesionVideos.txt"))
        p_pairs_les, p_vonly_les = _parse_video_keys(p_les_lines)
        p_pairs_non, p_vonly_non = _parse_video_keys(p_non_lines)

        videos = sorted(glob.glob(os.path.join(patient_dir, "v*")))
        for video_dir in videos:
            mv = re.search(r"v(\d+)$", os.path.basename(video_dir))
            if not mv:
                continue
            v_id = int(mv.group(1))

            # Decide label (priority: per-patient > global pairs > global v-only)
            if   (p_id, v_id) in p_pairs_les or v_id in p_vonly_les: y = 1
            elif (p_id, v_id) in p_pairs_non or v_id in p_vonly_non: y = 0
            elif (p_id, v_id) in g_pairs_les: y = 1
            elif (p_id, v_id) in g_pairs_non: y = 0
            elif v_id in g_vonly_les: y = 1
            elif v_id in g_vonly_non: y = 0
            else:
                unlabeled.append((p_id, v_id))
                continue

            inp = os.path.join(video_dir, "input")
            if not os.path.isdir(inp):
                no_input.append((p_id, v_id))
                continue

            fmap = _index_input_frames(inp)

            kf = _read_selected_frames(patient_dir, video_dir, p_id, v_id)
            if not kf:
                no_kf.append((p_id, v_id))
                continue

            for k in kf:
                fname = None
                # 1) by last-integer map
                fname = fmap["num_map"].get(k)
                # 2) zero-padded suffix
                if fname is None:
                    fname = fmap["pad_map"].get(f"_{k:05d}")
                # 3) exact expected stem p{p}_v{v}_{k:05d}
                if fname is None:
                    exp_stem = f"p{p_id}_v{v_id}_{k:05d}".lower()
                    fname = fmap["ext_map"].get(exp_stem)
                # 4) off-by-one fallback
                if fname is None:
                    for delta in (-1, +1):
                        alt = k + delta
                        if alt >= 0:
                            fname = fmap["num_map"].get(alt)
                            if fname: break
                # 5) brute suffix scan (png/jpg/jpeg)
                if fname is None:
                    suffix = f"_{k:05d}".lower()
                    for nm in fmap["names"]:
                        low = nm.lower()
                        if low.endswith(suffix + ".png") or low.endswith(suffix + ".jpg") or low.endswith(suffix + ".jpeg"):
                            fname = nm
                            break

                if fname is None:
                    missing_map.append((p_id, v_id, k))
                    continue

                img_path = os.path.join(inp, fname)
                if os.path.isfile(img_path):
                    rows.append({
                        "patient": p_id,
                        "video": v_id,
                        "frame": k,
                        "img_path": img_path,
                        "label": y
                    })

    df = pd.DataFrame(rows)
    if df.empty:
        msg = [
            "Index is empty (0 frames). Diagnostics:",
            f"- Base checked: {base}",
            f"- Patients found: {len(patients)}",
            f"- Global lesion/nonlesion lines: {len(g_les_lines)}/{len(g_non_lines)}",
            f"- Unlabeled videos (first 10): {unlabeled[:10]}",
            f"- Videos with no input/ (first 10): {no_input[:10]}",
            f"- Videos with no selectedFrames (first 10): {no_kf[:10]}",
            "Hints:",
            "  * Ensure per-patient lesion/nonlesion lists have entries like 'v1', 'v2', ...",
            "  * Ensure p{p}_v{v}_selectedFrames.txt exists in pX/ (or vY/).",
            "  * Check filenames in input/: we accept any pattern with digits; .png/.jpg/.jpeg.",
        ]
        raise RuntimeError("\n".join(msg))

    print(f"[build_index] frames: {len(df)} | patients: {df['patient'].nunique()} | videos: {df['video'].nunique()}")
    if unlabeled:
        print(f"[info] skipped unlabeled videos: {len(unlabeled)} (first 10: {unlabeled[:10]})")
    if no_kf:
        print(f"[info] videos without selectedFrames: {len(no_kf)} (first 10: {no_kf[:10]})")
    if no_input:
        print(f"[info] videos without input/: {len(no_input)} (first 10: {no_input[:10]})")
    # If you see a small missing_map count, it's fine; else investigate a sample
    from collections import Counter
    if 'missing_map' in locals() and missing_map:
        c = Counter((p,v) for p,v,_ in missing_map)
        print(f"[info] selected frames with no matching file: {len(missing_map)} (unique videos affected: {len(c)})")

    return df