import js from "@eslint/js";
import globals from "globals";
import importPlugin from "eslint-plugin-import";

export default [
  js.configs.recommended,

  {
    files: ["**/*.js"],
    languageOptions: {
      ecmaVersion: "latest",
      sourceType: "module",
      globals: {
        ...globals.node,
      },
    },
    plugins: {
      import: importPlugin,
    },
    rules: {
      // General best practices
      "no-unused-vars": ["error", { argsIgnorePattern: "^_" }],
      "no-console": "warn",
      "no-undef": "error",

      // Import rules (important for ESM backend)
      "import/no-unresolved": "error",
      "import/order": [
        "warn",
        {
          groups: [["builtin", "external", "internal"]],
          "newlines-between": "always",
        },
      ],

      // Prettier compatibility
      "arrow-body-style": "off",
      "prefer-arrow-callback": "off",
    },
  },
];