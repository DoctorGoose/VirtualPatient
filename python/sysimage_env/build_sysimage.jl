using Pkg

# Create or activate a dedicated environment for the system image
Pkg.activate("sysimage_env")  # You can specify any path or name you prefer

# Add the required packages to the environment
packages_to_add = [
    "DifferentialEquations",
    "PackageCompiler",
    "ThreadsX"
]

Pkg.add(packages_to_add)

# Update all packages
Pkg.update()

# Load/precompile/cache all packages_to_add
for pkg in packages_to_add
    @eval using $(Symbol(pkg))
end

# Now, import PackageCompiler
using PackageCompiler

# List of packages to include in the system image
packages = [
    "DifferentialEquations",
    "ThreadsX"
]

# Create the system image
create_sysimage(
    packages;
    sysimage_path="sysimage.so",
    incremental=false
)
