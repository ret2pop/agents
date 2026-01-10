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
        
        # 1. Initialize the poetry2nix toolset with your system's packages
        p2n = poetry2nix.lib.mkPoetry2Nix { inherit pkgs; };

        # 2. Extract the functions AND the default overrides from that initialized set
        inherit (p2n) mkPoetryEnv mkPoetryApplication defaultPoetryOverrides;

        # 3. Build the environment
        appEnv = mkPoetryEnv {
          projectDir = ./.;
          python = pkgs.python311;
          
          # Now 'defaultPoetryOverrides' is available here
          overrides = defaultPoetryOverrides.extend (self: super: {
            # Fixes for specific packages if needed
            # cryptography = super.cryptography.overridePythonAttrs (old: {
            #   buildInputs = (old.buildInputs or []) ++ [ pkgs.openssl ];
            # });
          });
        };

        app = mkPoetryApplication {
          projectDir = ./.;
          python = pkgs.python311;
        };

      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = [
            appEnv
            pkgs.poetry
            pkgs.ollama
          ];

          shellHook = ''
            echo "🔬 Epistemic Clean Room Active"
            echo "Python: $(python --version)"
          '';
        };

        packages.default = app;
      });
}
