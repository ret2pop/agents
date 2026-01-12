{
  description = "Deep Research Agent - Dynamic Epistemic Clean Room";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        
        # 1. DEFINE LIBRARIES NEEDED BY DYNAMIC WHEELS
        # These are the C libraries that PyPI packages (numpy, scipy, matplotlib) 
        # expect to find on a normal Linux system.
        runtimeLibs = with pkgs; [
          zlib
          stdenv.cc.cc.lib  # This is libstdc++
          glib
          libGL
          libxkbcommon
          fontconfig
          freetype
          expat
        ];
      in
      {
        devShells.default = pkgs.mkShell {
          # 2. PACKAGES
          # We include the libs in 'buildInputs' so they are available
          buildInputs = runtimeLibs;

          packages = [
            pkgs.python311
            pkgs.poetry
            pkgs.ollama
            # Add compilers so pip can build from source if a wheel is missing
            pkgs.gcc
            pkgs.gnumake 
          ];

          # 3. THE MAGIC FIX
          # We construct an LD_LIBRARY_PATH that points to the Nix store locations
          # of the libraries defined above.
          shellHook = ''
            export LD_LIBRARY_PATH=${pkgs.lib.makeLibraryPath runtimeLibs}:$LD_LIBRARY_PATH
            
            # Optional: configure poetry to keep the venv inside the project directory
            # This makes it easier to clean up if the agent messes it up.
            export POETRY_VIRTUALENVS_IN_PROJECT=true

            echo "🧪 Epistemic Clean Room (Dynamic Mode) Active"
            echo "----------------------------------------------"
            echo "Python: $(python --version)"
            echo "LD_LIBRARY_PATH: Patched for Numpy/Matplotlib support."
          '';
        };
      });
}
