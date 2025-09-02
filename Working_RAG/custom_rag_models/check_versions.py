import pkg_resources

# Read requirements.txt file
with open('requirements.txt', 'r') as f:
    required_packages = [line.strip() for line in f if line.strip() and not line.startswith('#')]

# Get the package names without version specifiers
package_names = [pkg.split('==')[0] for pkg in required_packages]

# Check installed versions
installed_versions = {}
for package in package_names:
    try:
        version = pkg_resources.get_distribution(package).version
        installed_versions[package] = version
    except pkg_resources.DistributionNotFound:
        installed_versions[package] = "Not installed"

# Print results
print("Installed package versions:")
for package, version in installed_versions.items():
    print(f"{package}=={version}")
