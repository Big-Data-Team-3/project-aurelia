"""
Validator DAG - Validates module imports and version requirements
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from airflow.providers.standard.operators.bash import BashOperator
import importlib
import importlib.metadata
import sys
from packaging import version
import re
import tomli


# Module name mappings (PyPI name -> import name)
MODULE_MAPPING = {
    "python-dotenv": "dotenv",
    "langchain-text-splitters": "langchain_text_splitters",
}


def parse_pyproject_dependencies():
    """Parse dependencies from pyproject.toml"""
    pyproject_path = "/Users/smatcha/Documents/BigData/project-aurelia/pyproject.toml"
    
    try:
        with open(pyproject_path, "rb") as f:
            data = tomli.load(f)
        
        dependencies = data.get("project", {}).get("dependencies", [])
        
        # Parse each dependency
        parsed_deps = []
        for dep in dependencies:
            # Parse format: "package>=version" or "package==version"
            match = re.match(r"([a-zA-Z0-9\-_]+)(>=|==|<=|>|<)?([\d.]+)?", dep)
            if match:
                pkg_name = match.group(1)
                operator = match.group(2) if match.group(2) else None
                req_version = match.group(3) if match.group(3) else None
                parsed_deps.append({
                    "package": pkg_name,
                    "operator": operator,
                    "required_version": req_version
                })
        
        return parsed_deps
    except Exception as e:
        print(f"Error parsing pyproject.toml: {e}")
        return []


def validate_imports(**context):
    """Validate that all required modules can be imported"""
    dependencies = parse_pyproject_dependencies()
    
    results = {
        "success": [],
        "failed": []
    }
    
    for dep in dependencies:
        pkg_name = dep["package"]
        # Get the actual import name (may differ from package name)
        import_name = MODULE_MAPPING.get(pkg_name, pkg_name.replace("-", "_"))
        
        try:
            # Attempt to import the module
            importlib.import_module(import_name)
            results["success"].append(pkg_name)
            print(f"✓ Successfully imported: {pkg_name} (as {import_name})")
        except ImportError as e:
            results["failed"].append({
                "package": pkg_name,
                "error": str(e)
            })
            print(f"✗ Failed to import: {pkg_name} (as {import_name}) - {e}")
    
    # Push results to XCom for downstream tasks
    context["ti"].xcom_push(key="import_results", value=results)
    
    print(f"\nImport Validation Summary:")
    print(f"  Successful: {len(results['success'])}")
    print(f"  Failed: {len(results['failed'])}")
    
    if results["failed"]:
        raise Exception(f"Failed to import {len(results['failed'])} module(s)")
    
    return results


def validate_versions(**context):
    """Validate that installed module versions meet requirements"""
    dependencies = parse_pyproject_dependencies()
    
    results = {
        "valid": [],
        "invalid": [],
        "not_found": []
    }
    
    for dep in dependencies:
        pkg_name = dep["package"]
        req_version = dep["required_version"]
        operator = dep["operator"]
        
        try:
            # Get installed version
            installed_version = importlib.metadata.version(pkg_name)
            
            if req_version and operator:
                # Compare versions
                installed_v = version.parse(installed_version)
                required_v = version.parse(req_version)
                
                is_valid = False
                if operator == ">=":
                    is_valid = installed_v >= required_v
                elif operator == "==":
                    is_valid = installed_v == required_v
                elif operator == "<=":
                    is_valid = installed_v <= required_v
                elif operator == ">":
                    is_valid = installed_v > required_v
                elif operator == "<":
                    is_valid = installed_v < required_v
                
                if is_valid:
                    results["valid"].append({
                        "package": pkg_name,
                        "installed": installed_version,
                        "required": f"{operator}{req_version}"
                    })
                    print(f"✓ {pkg_name}: {installed_version} satisfies {operator}{req_version}")
                else:
                    results["invalid"].append({
                        "package": pkg_name,
                        "installed": installed_version,
                        "required": f"{operator}{req_version}"
                    })
                    print(f"✗ {pkg_name}: {installed_version} does NOT satisfy {operator}{req_version}")
            else:
                # No version requirement specified
                results["valid"].append({
                    "package": pkg_name,
                    "installed": installed_version,
                    "required": "any"
                })
                print(f"✓ {pkg_name}: {installed_version} (no version constraint)")
                
        except importlib.metadata.PackageNotFoundError:
            results["not_found"].append(pkg_name)
            print(f"✗ {pkg_name}: Package not found")
    
    # Push results to XCom
    context["ti"].xcom_push(key="version_results", value=results)
    
    print(f"\nVersion Validation Summary:")
    print(f"  Valid: {len(results['valid'])}")
    print(f"  Invalid: {len(results['invalid'])}")
    print(f"  Not Found: {len(results['not_found'])}")
    
    if results["invalid"] or results["not_found"]:
        raise Exception(f"Version validation failed for {len(results['invalid']) + len(results['not_found'])} package(s)")
    
    return results


def generate_validation_report(**context):
    """Generate a comprehensive validation report"""
    ti = context["ti"]
    
    import_results = ti.xcom_pull(task_ids="validate_imports", key="import_results")
    version_results = ti.xcom_pull(task_ids="validate_versions", key="version_results")
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "python_version": sys.version,
        "import_validation": import_results,
        "version_validation": version_results,
        "overall_status": "PASSED"
    }
    
    # Determine overall status
    if (import_results and import_results.get("failed")) or \
       (version_results and (version_results.get("invalid") or version_results.get("not_found"))):
        report["overall_status"] = "FAILED"
    
    print("\n" + "="*60)
    print("VALIDATION REPORT")
    print("="*60)
    print(f"Status: {report['overall_status']}")
    print(f"Timestamp: {report['timestamp']}")
    print(f"Python Version: {sys.version.split()[0]}")
    print("="*60)
    
    return report


# Default DAG arguments
default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

# Define the DAG
with DAG(
    dag_id="validator_dag",
    default_args=default_args,
    description="Validates module imports and version requirements from pyproject.toml",
    schedule=None,  # Run manually
    start_date=datetime(2025, 10, 21),
    catchup=False,
    tags=["validation", "dependencies"],
) as dag:
    
    # Task 1: Validate imports
    validate_imports_task = PythonOperator(
        task_id="validate_imports",
        python_callable=validate_imports,
        # provide_context removed - context is automatically inferred
    )
    
    # Task 2: Validate versions
    validate_versions_task = PythonOperator(
        task_id="validate_versions",
        python_callable=validate_versions,
        # provide_context removed - context is automatically inferred
    )
    
    # Task 3: Generate report
    generate_report_task = PythonOperator(
        task_id="generate_validation_report",
        python_callable=generate_validation_report,
        # provide_context removed - context is automatically inferred
    )
    
    # Task 4: Print Python environment info
    print_env_task = BashOperator(
        task_id="print_environment_info",
        bash_command="""
        echo "Python Version: $(python --version)"
        echo "Python Path: $(which python)"
        echo "Pip Version: $(pip --version)"
        echo ""
        echo "Installed Packages:"
        pip list
        """
    )
    
    # Define task dependencies
    # Both import and version validation can run in parallel
    print_env_task >> [validate_imports_task, validate_versions_task] >> generate_report_task
