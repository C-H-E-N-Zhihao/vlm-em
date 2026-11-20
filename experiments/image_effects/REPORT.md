# Informe: efecto de la imagen en el Emergent Misalignment

> Plantilla. Se rellena con `results/image_effects/qwen2.5-32b-instruct/` (`primary.md`, `summary.md`,
> `checks.md`, `figures/`) cuando haya resultados. Siempre: EM con IC 95 %, valid, em_all y alineación media.

## Montaje
- Modelos: Qwen3-VL-32B (principal), InternVL3-38B (réplica). Qwen3-VL-4B solo para el piloto.
- Tareas: conspiracy (familia confirmatoria con 32B), careless_object y base (control negativo y comprobaciones).
- Juez: Qwen2.5-32B-Instruct en local (vLLM), con tag `qwen2.5-32b-instruct`, el mismo que zhihao/main. Prompts del paper. Los EM absolutos **no** se comparan con el paper (GPT-4o), solo las diferencias.
- Segunda opinión: Qwen2.5-72B, primarios: __.
- 10 muestras por pregunta, motor swift (RCP). Commit: __.
- Sanity check (`orig`, conspiracy): qwen3vl-4b __ %, qwen3vl-32b __ %, internvl3-38b __ % (paper con GPT-4o: 21,8 / 39,5 / 44,1).
- A/A (`orig_aa − orig`): Δ = __ pp [__, __], que es el ruido entre ejecuciones idénticas.

## Contrastes primarios (`primary.md`, Holm, familia confirmatoria qwen3vl-32b × conspiracy)
| # | Contraste | Δ EM [IC] | p Holm | Δ align | Δ valid | Réplica InternVL (signo) |
|---|---|---|---|---|---|---|
| 1 | noimage − orig | | | | | |
| 2 | mismatch − orig | | | | | |
| 3 | gray − noimage | | | | | |
| 4 | tendencia del ruido (pp/nivel) | | | | | |
| 5 | exp2b gray − full | | | | | |
| 6 | exp3 war − peace (p de permutación: __) | | | | | |
| 7 | exp4 photo related − plain | | | | | |

## Exp 1 — Ruido y blur
Figura `figures/exp1_visibility_<model>__<task>.png`. ¿Baja el EM al bajar la visibilidad? ¿Ruido y blur dan lo mismo con igual visibilidad? ¿Qué pasa con el ruido puro frente al gris?

## Exp 2a — ¿Tiene que encajar la imagen?
Figura `figures/exp2a_semantic.png`. ¿Qué parte del efecto de la imagen sobrevive a mismatch y a gray (fila `share` de `effects.csv`)? ¿Y con textpad − noimage (longitud)?

## Exp 2b — Objeto nombrado (unsafe-advice rate)
Figura `figures/exp2b_objects.png`. Relación con el reconocimiento (`checks.md`, yes/no) y con partial25 − lowres25 (¿parte del objeto o borrosidad?).

## Exp 3 — Imagen fija: paz / guerra / neutra
Figura `figures/exp3_fixed.png` (un punto por imagen). Comprobación del *mood* (`checks.md`). ¿war − peace por encima de la variación entre imágenes?

## Exp 4 — El texto se refiere o no a la imagen
Figura `figures/exp4_text.png`. related − plain con foto y con problema, y su DiD frente a sin imagen. Precisión en los problemas (`checks.md`): __ %.

## Respuestas inválidas y selección
`checks.md`: el desglose por condición. Qué contrastes llevan `!` (|Δvalid| > 5 pp).

## Conclusiones (lenguaje sencillo)

## Limitaciones
Juez local; 10 muestras; las imágenes del exp 2b son de entrenamiento; las fotos del exp 3 son reales y las de evaluación sintéticas; la longitud de textpad medida con el procesador HF.
