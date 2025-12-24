## Workflow para implementación de issues

Para cada issue, sigue este flujo de trabajo:

### 1. Preparación
- Asegúrate de estar en `main` actualizado: `git checkout main && git pull`
- Crea la branch con formato: `feature/<issue_number>-<short-description>`
    - Ejemplo: `feature/42-add-greeks-calculation`

### 2. Implementación
- Desarrolla el código necesario para resolver el issue
- Mantén commits atómicos con mensajes descriptivos siguiendo conventional commits:
    - `feat(module): add delta calculation for european options`
    - `fix(risk): correct margin requirement formula`
- Actualiza `src/lib.rs`:
    - Documenta las nuevas APIs públicas con doc-comments (`//!`)
    - Añade ejemplos de uso en la documentación cuando sea apropiado
    - Exporta los nuevos módulos/tipos si corresponde

### 3. Verificación
- Ejecuta `make lint-fix pre-push`
- Si hay errores, corrígelos antes de continuar
- Verifica que todos los tests pasan: `cargo test`

### 4. Push y PR
- Push de la branch: `git push -u origin HEAD`
- Crea el PR: `gh pr create --base main --fill`
    - Asegúrate de que el título referencia el issue: `feat: add greeks calculation (#42)`
    - En el body incluye `Closes #<issue_number>` para auto-cerrar el issue

### 5. Post-PR
- Espera a que pasen los checks de CI
- Si fallan, corrige y pushea los cambios
- Una vez el PR es mergeado, actualiza la branch `main` y borra la branch de la feature: `git checkout main && git pull && git branch -d feature/<issue_number>-<short-description>`
