import subprocess
import sys
import textwrap


def test_kernel_load_in_subprocess():
    code = textwrap.dedent(
        """
        import sys
        from certificates import verified_kernel as vk
        try:
            kernel = vk.load_kernel(strict=False)
            print("LOADED:", bool(kernel))
            sys.exit(0)
        except Exception as exc:
            print("EXC:", exc, file=sys.stderr)
            sys.exit(2)
        """
    )
    process = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert (
        process.returncode == 0
    ), f"Kernel load subprocess failed: rc={process.returncode}\nSTDOUT:\n{process.stdout}\nSTDERR:\n{process.stderr}"
