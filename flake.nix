{
  description = "Deep Research Agent - Epistemic Clean Room";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    poetry2nix = {
      url = "github:nix-community/poetry2nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { self, nixpkgs, flake-utils, poetry2nix }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        
        # 1. SETUP
        # Use Python 3.11 (Standard for AI/Data Science stability)
        # Avoids conflicts with system tools that might rely on 3.12
        myPython = pkgs.python311; 

        p2n = poetry2nix.lib.mkPoetry2Nix { inherit pkgs; };
        inherit (p2n) mkPoetryEnv mkPoetryApplication defaultPoetryOverrides;

        # 2. BUILD THE VIRTUAL ENV
        appEnv = mkPoetryEnv {
          projectDir = ./.;
          python = myPython;
          preferWheels = true; # Speed up builds significantly
          
          overrides = defaultPoetryOverrides.extend (self: super: {
            # Add overrides here if specific packages fail to build
          });
        };

        # 3. BUILD THE APP (Optional)
        app = mkPoetryApplication {
          projectDir = ./.;
          python = myPython;
        };

      in
      {
        devShells.default = pkgs.mkShell {
          # 'packages' adds tools to the PATH. 
          # appEnv goes FIRST to ensure 'python' resolves to your env, not Poetry's internal python.
          packages = [ 
            appEnv
            pkgs.poetry
            pkgs.ollama
          ];

          # Clean up environment variables to prevent conflicts
          shellHook = ''
            # Unset PYTHONPATH so we don't accidentally load system libs
            unset PYTHONPATH
            
            echo "🔬 Epistemic Clean Room Active"
            echo "--------------------------------"
            echo "Python Location: $(which python)"
            echo "Python Version:  $(python --version)"
            echo "Environment:     Native Nix (No .venv required)"
          '';
        };

        packages.default = app;
      });
}
