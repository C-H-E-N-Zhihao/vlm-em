# Informe: efecto de la imagen en el Emergent Misalignment

> Plantilla. Se rellena con `results/image_effects/summary.md` y `figures/` tras ejecutar el grid.

## Montaje
- Modelos / tareas / backend / nº de muestras:
- Sanity check (`orig`, qwen3vl-4b, ordinary_scene_conspiracy): EM = __ % (paper 21,8 %)

## Exp 1 — Ruido
Figura: `figures/exp1_degradation.png` (ruido y blur; arriba EM, abajo valid y visibilidad). ¿Baja el EM al bajar la visibilidad? ¿Ruido y blur dan lo mismo?

## Exp 2a — ¿Tiene que encajar la imagen con la pregunta?
Figura: `figures/exp2a_semantic.png`. orig vs mismatch vs gris vs sin imagen.

## Exp 2b — Objeto nombrado (completo / parcial / ausente)
Figura: `figures/exp2b_objects.png`. Además: ¿qué objeto dice ver el modelo? (visibilidad)

## Exp 3 — Imagen fija: paz / guerra / neutra
Figura: `figures/exp3_fixed.png`.

## Exp 4 — Imagen fija, texto relacionado vs no relacionado
Figura: `figures/exp4_text.png`. Precisión en los problemas: __ %.

## Contrastes primarios
Tabla `primary.md` (Holm). El resto es exploratorio.

## Conclusiones (lenguaje sencillo)

## Limitaciones
