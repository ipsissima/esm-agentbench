import subprocess

from certificates.verified_kernel import load_kernel


def test_kernel_load_and_smoke():
    try:
        load_kernel(strict=True)
    except Exception as exc:
        print("Kernel load failed:", exc)
        print("=== versions ===")
        for cmd in (("coqc", "--version"), ("ocamlopt", "-version"), ("opam", "list")):
            try:
                out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
                print(cmd, "=>", out.splitlines()[0] if out else out)
            except Exception as ex:
                print("Failed to run", cmd, ":", ex)
        raise
