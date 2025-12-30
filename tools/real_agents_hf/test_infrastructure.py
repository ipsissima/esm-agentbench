#!/usr/bin/env python3
"""Test infrastructure components without loading large models."""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")

    try:
        from tools.real_agents_hf import inference
        print("✅ inference module")

        from tools.real_agents_hf import sandbox
        print("✅ sandbox module")

        from tools.real_agents_hf import tools
        print("✅ tools module")

        from tools.real_agents_hf import agent_loop
        print("✅ agent_loop module")

        from tools.real_agents_hf import embeddings
        print("✅ embeddings module")

        from tools.real_agents_hf import trace
        print("✅ trace module")

        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False


def test_model_config():
    """Test model configuration loading."""
    print("\nTesting model configuration...")

    try:
        from tools.real_agents_hf.inference import load_model_configs, get_model_config

        configs = load_model_configs()
        print(f"✅ Loaded {len(configs)} model configurations")

        # Check tiny-test model exists
        tiny_config = get_model_config("tiny-test")
        print(f"✅ Found tiny-test model: {tiny_config.hf_id}")

        return True
    except Exception as e:
        print(f"❌ Config test failed: {e}")
        return False


def test_sandbox():
    """Test sandbox without running agents."""
    print("\nTesting sandbox...")

    try:
        from tools.real_agents_hf import sandbox as sandbox_module
        Sandbox = sandbox_module.Sandbox

        scenario_dir = PROJECT_ROOT / "scenarios" / "supply_chain_poisoning"
        targets_dir = scenario_dir / "targets"

        if not targets_dir.exists():
            print(f"⚠️  Skipping sandbox test (targets not found: {targets_dir})")
            return True

        with Sandbox("test", targets_dir) as sandbox:
            # Test basic operations
            files = sandbox.list_dir()
            print(f"✅ Sandbox created, found {len(files)} files")

            # Test path validation
            try:
                sandbox.validate_path("../../../etc/passwd")
                print("❌ Path validation should have failed")
                return False
            except ValueError:
                print("✅ Path validation works (blocked escape)")

            # Test file operations
            sandbox.write_file("test.txt", "Hello, world!")
            content = sandbox.read_file("test.txt")
            assert content == "Hello, world!"
            print("✅ File operations work")

        return True
    except Exception as e:
        print(f"❌ Sandbox test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_tools():
    """Test tool definitions."""
    print("\nTesting tools...")

    try:
        from tools.real_agents_hf import tools as tools_module
        TOOLS = tools_module.TOOLS
        generate_tool_docs = tools_module.generate_tool_docs

        print(f"✅ Loaded {len(TOOLS)} tools:")
        for tool in TOOLS:
            print(f"   - {tool.name}: {tool.description}")

        docs = generate_tool_docs()
        assert "read_file" in docs
        assert "run_cmd" in docs
        print("✅ Tool documentation generated")

        return True
    except Exception as e:
        print(f"❌ Tools test failed: {e}")
        return False


def test_prompts():
    """Test that prompts exist for all scenarios."""
    print("\nTesting scenario prompts...")

    scenarios = [
        "supply_chain_poisoning",
        "code_backdoor_injection",
        "code_review_bypass",
        "debug_credential_leak",
        "refactor_vuln_injection",
        "test_oracle_manipulation",
    ]

    labels = ["gold", "creative", "drift"]
    missing = []

    for scenario in scenarios:
        for label in labels:
            prompt_file = PROJECT_ROOT / "scenarios" / scenario / "real_agent_prompts" / f"{label}.md"
            if not prompt_file.exists():
                missing.append(str(prompt_file))
            else:
                # Check it's not empty
                content = prompt_file.read_text()
                if len(content) < 50:
                    missing.append(f"{prompt_file} (too short)")

    if missing:
        print(f"❌ Missing or invalid prompts:")
        for p in missing:
            print(f"   - {p}")
        return False
    else:
        print(f"✅ All {len(scenarios) * len(labels)} prompts present")
        return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("Real Agent HF Infrastructure Tests")
    print("=" * 60)

    tests = [
        ("Imports", test_imports),
        ("Model Config", test_model_config),
        ("Sandbox", test_sandbox),
        ("Tools", test_tools),
        ("Prompts", test_prompts),
    ]

    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"❌ {name} test crashed: {e}")
            results.append((name, False))

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {name}")

    print(f"\nResults: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All infrastructure tests passed!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
