{
  description = "Deep Research Agent - Dynamic Epistemic Clean Room";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    poetry2nix.url = "github:nix-community/poetry2nix";
    poetry2nix.inputs.nixpkgs.follows = "nixpkgs";
  };

  outputs = { self, nixpkgs, flake-utils, poetry2nix }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        inherit (poetry2nix.lib.mkPoetry2Nix { inherit pkgs; }) mkPoetryEnv;
        
        myPythonEnv = mkPoetryEnv {
          projectDir = ./.;
          python = pkgs.python311;
          preferWheels = true;
        };
      in
      {
        devShells.default = pkgs.mkShell {
          packages = [
            myPythonEnv
            pkgs.poetry
            pkgs.ollama
          ];

          shellHook = ''
            echo "🧪 Epistemic Clean Room (poetry2nix Mode) Active"
          '';
        };
      });
}
