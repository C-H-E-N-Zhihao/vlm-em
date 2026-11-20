# Informe: efecto de la imagen en el Emergent Misalignment

> Plantilla. Se rellena con `results/image_effects/summary.md` y `figures/` tras ejecutar el grid.

## Montaje
- Modelos: Qwen3-VL-32B (principal), InternVL3-38B (2ª familia, el de Zhihao); Qwen3-VL-4B solo piloto
- Tareas / backend (swift, RCP) / nº de muestras (10 por pregunta):
- Juez: Qwen2.5-32B-Instruct local (vLLM), tag `qwen2.5-32b-instruct` — el mismo que zhihao/main.
  Los prompts son los del paper, pero el juez no: **los EM absolutos no se comparan con el paper (GPT-4o)**,
  solo las diferencias entre condiciones y con los resultados de Zhihao.
- Sanity check (`orig`, ordinary_scene_conspiracy): qwen3vl-4b EM = __ %, qwen3vl-32b EM = __ %,
  internvl3-38b EM = __ % (paper con GPT-4o: 21,8 / 39,5 / 44,1 %; mismo orden de magnitud esperado)

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
