import subprocess


def _print_versions():
    for cmd in (("coqc", "--version"), ("ocamlopt", "-version"), ("opam", "version")):
        try:
            out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
            print(cmd, "=>", out.splitlines()[0])
        except Exception as ex:
            print("Failed to run", cmd, ":", ex)


def test_kernel_load_and_smoke():
    from certificates.verified_kernel import load_kernel

    try:
        load_kernel(strict=True)
    except Exception as e:
        print("Kernel load failed:", e)
        _print_versions()
        try:
            from pathlib import Path

            p = Path("UELAT") / "kernel_verified.so"
            if p.exists():
                print("kernel file:", p, "size:", p.stat().st_size)
        except Exception:
            pass
        raise

    assert True
