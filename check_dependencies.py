#!/usr/bin/env python3
"""
Comprehensive dependency checker for RunPod Chatterbox
Checks all dependencies before building to prevent runtime failures
"""

import subprocess
import sys
import importlib
import pkg_resources
from typing import Dict, List, Tuple

def run_command(cmd: List[str]) -> Tuple[int, str, str]:
    """Run a command and return exit code, stdout, stderr"""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        return result.returncode, result.stdout, result.stderr
    except Exception as e:
        return 1, "", str(e)

def check_package_installed(package_name: str) -> Tuple[bool, str]:
    """Check if a package is installed and return version"""
    try:
        version = pkg_resources.get_distribution(package_name).version
        return True, version
    except pkg_resources.DistributionNotFound:
        return False, "Not installed"

def check_import_works(module_name: str) -> Tuple[bool, str]:
    """Check if a module can be imported successfully"""
    try:
        importlib.import_module(module_name)
        return True, "Import successful"
    except ImportError as e:
        return False, f"Import failed: {e}"
    except Exception as e:
        return False, f"Other error: {e}"

def check_dependency_chain() -> Dict[str, Dict]:
    """Check the full dependency chain that RunPod needs"""
    
    print("🔍 ===== DEPENDENCY CHAIN CHECK =====")
    
    # Define the critical dependency chain
    dependency_chain = {
        "runpod": {
            "import_name": "runpod",
            "dependencies": ["aiohttp", "multidict", "yarl", "propcache"]
        },
        "aiohttp": {
            "import_name": "aiohttp",
            "dependencies": ["yarl", "multidict", "propcache"]
        },
        "yarl": {
            "import_name": "yarl",
            "dependencies": ["propcache"]
        },
        "chatterbox": {
            "import_name": "chatterbox",
            "dependencies": ["torch", "transformers", "diffusers", "librosa", "numpy"]
        }
    }
    
    results = {}
    
    for package, info in dependency_chain.items():
        print(f"\n📦 Checking {package}...")
        
        # Check if package is installed
        installed, version = check_package_installed(package)
        print(f"   📋 Installed: {installed} (version: {version})")
        
        # Check if import works
        import_works, import_msg = check_import_works(info["import_name"])
        print(f"   🔄 Import: {import_works} - {import_msg}")
        
        # Check dependencies
        dep_results = {}
        for dep in info["dependencies"]:
            dep_installed, dep_version = check_package_installed(dep)
            dep_import, dep_import_msg = check_import_works(dep)
            dep_results[dep] = {
                "installed": dep_installed,
                "version": dep_version,
                "import_works": dep_import,
                "import_msg": dep_import_msg
            }
            print(f"   └─ {dep}: installed={dep_installed} (v{dep_version}), import={dep_import}")
        
        results[package] = {
            "installed": installed,
            "version": version,
            "import_works": import_works,
            "import_msg": import_msg,
            "dependencies": dep_results
        }
    
    return results

def check_version_compatibility() -> Dict[str, Dict]:
    """Check version compatibility issues"""
    
    print("\n🔍 ===== VERSION COMPATIBILITY CHECK =====")
    
    compatibility_checks = {
        "numpy": {
            "min_version": "1.25.2",
            "max_version": "2.6.0",
            "reason": "Required for SciPy compatibility"
        },
        "huggingface_hub": {
            "min_version": "0.23.2",
            "reason": "Required for transformers compatibility"
        },
        "torch": {
            "expected_version": "2.6.0",
            "reason": "Must match forked repository requirements"
        }
    }
    
    results = {}
    
    for package, requirements in compatibility_checks.items():
        print(f"\n📦 Checking {package} version compatibility...")
        
        installed, version = check_package_installed(package)
        
        if not installed:
            print(f"   ❌ {package} not installed")
            results[package] = {"status": "missing", "version": None}
            continue
        
        print(f"   📋 Installed version: {version}")
        
        # Check version constraints
        if "min_version" in requirements:
            try:
                if pkg_resources.parse_version(version) < pkg_resources.parse_version(requirements["min_version"]):
                    print(f"   ❌ Version {version} < {requirements['min_version']}")
                    results[package] = {"status": "too_old", "version": version, "required": requirements["min_version"]}
                    continue
            except Exception as e:
                print(f"   ⚠️ Could not parse version: {e}")
        
        if "max_version" in requirements:
            try:
                if pkg_resources.parse_version(version) >= pkg_resources.parse_version(requirements["max_version"]):
                    print(f"   ❌ Version {version} >= {requirements['max_version']}")
                    results[package] = {"status": "too_new", "version": version, "required": f"<{requirements['max_version']}"}
                    continue
            except Exception as e:
                print(f"   ⚠️ Could not parse version: {e}")
        
        if "expected_version" in requirements:
            if version != requirements["expected_version"]:
                print(f"   ⚠️ Version mismatch: {version} != {requirements['expected_version']}")
                results[package] = {"status": "mismatch", "version": version, "expected": requirements["expected_version"]}
            else:
                print(f"   ✅ Version matches expected: {version}")
                results[package] = {"status": "ok", "version": version}
        else:
            print(f"   ✅ Version compatible: {version}")
            results[package] = {"status": "ok", "version": version}
    
    return results

def check_chatterbox_repository() -> Dict[str, any]:
    """Check if we're using the forked repository"""
    
    print("\n🔍 ===== CHATTERBOX REPOSITORY CHECK =====")
    
    try:
        import chatterbox
        print(f"✅ chatterbox module loaded from: {chatterbox.__file__}")
        
        # Check for voice profile methods
        from chatterbox.tts import ChatterboxTTS
        
        has_load_profile = hasattr(ChatterboxTTS, 'load_voice_profile')
        has_save_profile = hasattr(ChatterboxTTS, 'save_voice_profile')
        
        print(f"📋 Voice profile methods:")
        print(f"   └─ load_voice_profile: {'✅' if has_load_profile else '❌'}")
        print(f"   └─ save_voice_profile: {'✅' if has_save_profile else '❌'}")
        
        # Check pip info
        code, stdout, stderr = run_command(['pip', 'show', 'chatterbox-tts'])
        if code == 0:
            print(f"📦 Package info:\n{stdout}")
            
            # Check if it's from forked repo
            if 'chrijaque' in stdout.lower():
                repo_status = "forked"
                print("✅ Package appears to be from forked repository")
            elif 'git' in stdout.lower():
                repo_status = "git"
                print("⚠️ Package appears to be from git installation")
            else:
                repo_status = "pypi"
                print("❌ Package appears to be from PyPI (original repository)")
        else:
            repo_status = "unknown"
            print(f"❌ Could not get package info: {stderr}")
        
        return {
            "module_path": chatterbox.__file__,
            "has_load_profile": has_load_profile,
            "has_save_profile": has_save_profile,
            "repository": repo_status,
            "pip_info": stdout if code == 0 else None
        }
        
    except Exception as e:
        print(f"❌ Failed to check chatterbox: {e}")
        return {"error": str(e)}

def main():
    """Run all dependency checks"""
    
    print("🚀 ===== RUNPOD CHATTERBOX DEPENDENCY CHECKER =====")
    print("This will check all dependencies before building to prevent runtime failures\n")
    
    # Check dependency chain
    chain_results = check_dependency_chain()
    
    # Check version compatibility
    version_results = check_version_compatibility()
    
    # Check chatterbox repository
    chatterbox_results = check_chatterbox_repository()
    
    # Summary
    print("\n" + "="*60)
    print("📊 ===== SUMMARY =====")
    
    # Count issues
    issues = []
    
    # Check dependency chain issues
    for package, info in chain_results.items():
        if not info["import_works"]:
            issues.append(f"❌ {package} import failed: {info['import_msg']}")
        
        for dep, dep_info in info["dependencies"].items():
            if not dep_info["import_works"]:
                issues.append(f"❌ {package} dependency {dep} import failed: {dep_info['import_msg']}")
    
    # Check version issues
    for package, info in version_results.items():
        if info["status"] != "ok":
            issues.append(f"❌ {package} version issue: {info['status']} (v{info.get('version', 'unknown')})")
    
    # Check chatterbox issues
    if "error" in chatterbox_results:
        issues.append(f"❌ chatterbox check failed: {chatterbox_results['error']}")
    elif not chatterbox_results.get("has_load_profile", False) or not chatterbox_results.get("has_save_profile", False):
        issues.append("❌ chatterbox missing voice profile methods")
    elif chatterbox_results.get("repository") == "pypi":
        issues.append("❌ chatterbox from PyPI instead of forked repository")
    
    if issues:
        print("❌ ISSUES FOUND:")
        for issue in issues:
            print(f"   {issue}")
        print(f"\n🔧 Total issues: {len(issues)}")
        return 1
    else:
        print("✅ ALL CHECKS PASSED - Ready for build!")
        return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 