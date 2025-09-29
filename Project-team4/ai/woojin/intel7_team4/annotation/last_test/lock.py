# lock.py
import fcntl
import sys
import os

LOCKFILE = "/tmp/neck_eye_processed.lock"

def acquire_single_instance_lock():
    fp = open(LOCKFILE, "w")
    try:
        fcntl.flock(fp, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        fp.close()
        print("[LOCK] another instance is running. exit.")
        sys.exit(1)
    fp.write(str(os.getpid()))
    fp.flush()
    print(f"[LOCK] acquired ({LOCKFILE}), pid={os.getpid()}")
    return fp

def release_single_instance_lock(fp):
    try:
        fcntl.flock(fp, fcntl.LOCK_UN)
        fp.close()
        try:
            os.remove(LOCKFILE)
        except Exception:
            pass
        print("[LOCK] released")
    except Exception:
        pass
