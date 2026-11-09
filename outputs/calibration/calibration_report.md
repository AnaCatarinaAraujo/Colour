# Calibração de cor com cartões Pantone — relatório

Gerado por `src/run_calibration.py` (branch `colour-calibration`). Os dados originais não foram alterados; tudo o que está aqui está em `outputs/calibration/`.

## 1. Cartões usados
- **32 cartões fotografados, 32 segmentados, 32 usados.** Todos têm correspondência por código com o Excel `Calibration Curve Sophia.xlsx` (colunas `L*`, `a*`, `b*`; as colunas "(Matlab)" foram ignoradas). Não há cartões sem correspondência.
- O código **18R7 aparece duas vezes** no Excel (L\* 40,47 / 41,29); usou-se a média das duas leituras.
- Medição: o mesmo L\*a\*b\* do script do grão (`extract_color_values`: OpenCV `BGR2LAB` em 8 bits; L\*·100/255, a\*−128, b\*−128). Segmentação: Otsu no canal V, componente mais próxima do centro (exclui outros cartões nos bordos), preenchimento de furos e erosão de 4% do lado menor do cartão (≈ 25–35 px). Imagens de verificação em `masks/` (vermelho = cartão, verde = zona medida).
- O DP por pixel dentro de cada cartão é 0,3–1,0 unidades, exceto 08G5, 08G7 e 03G7 (DP de L\* 2–3,6), que têm uma ligeira sombra no bordo inferior.

## 2. Modelos testados e validação (ΔE\*ab, CIE76; n = 32)

| Modelo | Parâmetros | Ajuste: médio / máx | **LOO: médio / máx** | Sobreajuste? |
|---|---|---|---|---|
| Sem calibração | 0 | 14,7 / 22,5 | 14,7 / 22,5 | — |
| (a) Linear por canal (L\*a\*b\*) | 6 | 8,8 / 15,0 | 9,4 / 16,5 | não |
| **(b) Afim 3×3 + termo independente (L\*a\*b\*)** | 12 | **5,4 / 10,1** | **6,2 / 12,1** | **não** |
| Afim + L\*² | 15 | 5,2 / 10,5 | 6,2 / 12,7 | sim (instável) |
| Afim + L\*² + L\*³ | 18 | 5,0 / 10,6 | 6,3 / 13,0 | sim (instável) |
| Polinómio completo 2.ª ordem (L\*a\*b\*) | 30 | 3,7 / 8,2 | 6,0 / 16,6 | **sim** |
| RGB linear → XYZ, matriz 3×3 | 9 | 7,1 / 31,0 | 8,1 / 43,7 | não |
| RGB linear → XYZ, root-polynomial | 18 | 6,7 / 15,6 | 9,9 / 50,9 | não |
| RGB linear → XYZ, polinómio 2.ª ordem | 30 | 4,9 / 13,3 | 9,2 / 60,5 | sim |

Critério de sobreajuste: erro LOO > 1,5 × erro de ajuste, **ou** a cor prevista para alguma fotografia de grão muda mais de 3 ΔE quando se retira um único cartão (`calibration_model_comparison.csv`).

## 3. Modelo escolhido: afim 3×3 + termo independente em L\*a\*b\*

    L*cal = 0,9281·L* + 0,0741·a* + 0,2425·b* − 7,414
    a*cal = 0,0493·L* + 1,1865·a* + 0,0081·b* − 4,120
    b*cal = 0,1121·L* + 0,3007·a* + 1,1389·b* − 5,213

(L\*, a\*, b\* = valores medidos pelo script; coeficientes em `model_selected.json`; função `apply_calibration()` em `src/colour/calibration.py`.)

**Porquê este e não o polinomial:** o polinómio de 2.ª ordem tem o menor erro LOO médio (6,0 contra 6,2), mas a diferença é desprezável e o modelo mostra sobreajuste claro: o erro sobe 61% do ajuste para o LOO, o máximo sobe para 16,6, e retirar um só cartão muda a cor prevista do grão até 6,5 ΔE. Os modelos polinomiais mais pequenos (com L\*² e L\*³) não melhoram o LOO e também são instáveis no grão. Os modelos em RGB linear falham nos azuis saturados (06B7, canal B perto de 255). O afim é o modelo com menor erro LOO entre os que são estáveis.

## 4. Grão: unidade de medição e aplicação
- **Unidade = uma fotografia** (média de todos os pixels de grão segmentados na fotografia, exatamente como em `color_summary_by_image.csv`). O desvio-padrão é entre fotografias (n−1).
- **Significado do DP depende do tratamento** (a mesma estrutura da análise de textura):
  - **Soaked, Im.1–Im.3, Boiled, ImT.1–ImT.3:** cada fotografia é uma **réplica independente** (lote diferente). O DP é a variabilidade entre lotes e estes tratamentos podem entrar em testes estatísticos (por exemplo, ANOVA + Tukey).
  - **Sol.1–Sol.7 (cozedura solar): sem réplicas independentes.** As 3 fotografias de cada dia são do mesmo lote, por isso o DP mostra só a variação dentro do lote e subestima a variabilidade real (daí valores de 0,0–0,6). Estes tratamentos devem ser apresentados de forma descritiva (média e ΔE vs Soaked) e **não devem entrar em testes de significância** como se as fotografias fossem réplicas (seria pseudorreplicação).
  - As tabelas têm a coluna "DP refere-se a" com esta indicação para cada tratamento.
- Cada fotografia foi contada **uma vez** (o `main.py` original conta cada ficheiro duas vezes no Windows).
- A calibração é aplicada a cada fotografia **antes** de calcular as médias. C\* e h° são calculados por fotografia a partir dos a\*, b\* calibrados. O ΔE\* vs Soaked usa as médias de L\*, a\* e b\*.
- Correspondência ficheiro → tratamento: `src/treatment_mapping.csv` (66 fotografias, todas encontradas).
- Tabelas: `treatment_summary_calibrated.csv`, `treatment_summary_uncalibrated.csv` e `treatment_summary.xlsx` (com folhas numéricas e a comparação de modelos). Medições por fotografia: `grain_measurements_*.csv`.

## 4b. Estatística entre tratamentos (`src/stats_treatments.py`, resultados em `stats/`)
- **Dados:** valores por fotografia (calibrados; repetido também para os não calibrados). Só os tratamentos com réplicas independentes: Soaked (n = 7), Im.1–Im.3 (3 cada), Boiled (n = 20), ImT.1–ImT.3 (3 cada); k = 8, N = 45. **Sol.1–Sol.7 ficam fora dos testes** e são marcados com **†** nas tabelas, em vez de letras.
- **Testes:** Levene (centro = média) para homogeneidade de variâncias; ANOVA de Welch (não assume variâncias iguais); post-hoc de Games-Howell (variâncias e n desiguais); letras pelo algoritmo insert-absorb a partir dos p do Games-Howell (α = 0,05). Se a ANOVA de Welch não for significativa, todos os tratamentos ficam com "a".
- **Resultados (calibrados):** as variâncias não são homogéneas para L\* (Levene p = 0,038), a\* (p = 0,023) e h° (p = 0,001), o que justifica usar Welch + Games-Howell; b\* (p = 0,066) e C\* (p = 0,079) não rejeitam a homogeneidade. A ANOVA de Welch é significativa para todas as variáveis: L\*, b\*, C\* p < 0,001; a\* p < 0,001; h° p = 0,008.
- **Verificação:** os testes foram implementados com `scipy` e comparados com o pacote `pingouin`: os p de Levene, Welch e Games-Howell coincidem (diferenças < 10⁻¹³). As letras foram verificadas contra todos os pares (nenhuma incoerência).
- **Limitação:** com n = 3 em seis dos oito grupos, a potência do Games-Howell é baixa (df pequenos). Diferenças reais moderadas podem não sair significativas, sobretudo em a\* e h°.
- Ficheiros: `stats/treatment_stats.xlsx` (tabelas com letras e †, Levene/Welch, Games-Howell), `stats/levene_welch.csv`, `stats/games_howell.csv`, `stats/treatment_summary_*_letters.csv`, `stats/NOTA.txt` (nota de rodapé para as tabelas).

## 4c. Figura 5: imagens representativas (`src/make_figure5.py`, saídas em `outputs/figures/`)
- Uma fotografia por tipo de tratamento: Soaked, Boiled, Im.3 (imitação sem temperatura), ImT.2 (imitação com temperatura) e Sol.3 (cozedura solar). Em cada um, a de menor ΔE\*ab em relação à média do seu tratamento em L\*a\*b\* calibrado (a do Im.3, `AmbT3.13.05.26`, foi indicada pela autora e é também a mais próxima, ΔE 0,2). Os nomes e os ΔE estão em `outputs/figures/Figure5_images_panels.txt`.
- **As fotografias mostradas na figura NÃO têm calibração de cor**: são imagens ilustrativas. A calibração aplica-se só aos valores numéricos L\*a\*b\*.
- A máscara é a mesma da análise de cor (`segment_chickpeas`). Todas as imagens foram reamostradas para a mesma ampliação (ver o ponto 8 dos problemas) e recortadas com o mesmo tamanho, centradas no grão.
- Não há barra de escala, porque nenhuma fotografia tem uma referência de tamanho conhecida.
- Gera-se com `python src/make_figure5.py --use Im.3=AmbT3.13.05.26.JPG --crop-size 1140` (15 painéis, (a)–(o)). Numa versão anterior entravam também Sol.1 e Sol.6; foram retiradas para simplificar a figura.
- A coluna "Mask contour" desenha o contorno exterior e também os buracos da máscara (`cv2.RETR_CCOMP`), para que os espaços entre grãos encostados apareçam fora do contorno. Isto muda só o desenho da figura; a segmentação e os resultados ficam iguais.

## 4d. Pixels escuros incluídos pelo fecho morfológico (`src/analysis_dark_pixels.py`, saídas em `dark_pixels/`)
- Para cada fotografia calculou-se a percentagem de pixels da máscara com V abaixo do limiar de Otsu (só o fecho os pode incluir) e o L\*a\*b\* calibrado com e sem esses pixels. O pipeline não foi alterado; com todos os pixels, os valores coincidem exatamente com os do pipeline.
- **Os pixels escuros são 0,04–0,27% da máscara** (média 0,13%). Excluí-los muda a cor em **ΔE\* ≤ 0,10** (média 0,04; o máximo é no `boild_3_08.04.26`). O efeito é desprezável em todos os tratamentos (`dark_pixels.xlsx`: tabela por fotografia e máximo por tratamento).

## 5. Problemas encontrados
1. **Não há cartões perto da cor do grão.** O grão medido tem L\* 39–55, a\* 6–10, b\* 23–33. O cartão mais próximo (30A7) está a 22 ΔE, e os seguintes a mais de 30. Os valores do grão ficam por isso numa zona sem cartões, onde o modelo interpola ou extrapola. Os modelos razoáveis (afim, RGB linear, root-polynomial) concordam a 3–4 ΔE nas médias por tratamento, mas o polinomial difere 11–13 ΔE (`sensitivity_vs_selected_model.csv`). **Os valores absolutos calibrados têm por isso uma incerteza de alguns ΔE; as diferenças entre tratamentos (ΔE vs Soaked) são mais robustas**: com o modelo afim mudam no máximo ≈ 1,7 unidades em relação aos valores não calibrados, e os extremos mantêm-se (Sol.6 o mais diferente do Soaked; Sol.3 e ImT.1 os mais próximos). Já a ordem dos tratamentos intermédios (ΔE 3–5) muda, por isso essas diferenças não devem ser sobreinterpretadas. Recomendação: fotografar e medir no colorímetro 5–10 cartões bege/amarelo-acastanhados (perto de L\* 45–55, a\* 5–12, b\* 25–40) e recalibrar.
2. **Erro residual elevado (ΔE LOO ≈ 6).** A resposta da câmara não é linear em L\*: nos cartões escuros o L\* medido fica 10–20 unidades acima da referência, nos claros fica quase igual, e alguns cartões claros estão perto da saturação (R ≈ 254). Com 32 cartões, os modelos não lineares não conseguem corrigir isto sem sobreajuste. Se as fotos `.CR2` (RAW) estiverem disponíveis para todas as sessões e cartões, uma calibração a partir do RAW linear seria mais rigorosa.
3. **Iluminante e observador do colorímetro.** O Konica Minolta CR-400 mede com observador de 2° e iluminante C ou D65 (configurável). A calibração leva diretamente para o espaço L\*a\*b\* do colorímetro, por isso os valores calibrados ficam com o iluminante que estava configurado. **Ainda falta confirmar se era C ou D65** (está no menu de configuração do aparelho ou na impressão/exportação das leituras).
4. **Ficheiros e nomes:** o `.gitignore` do repositório contém `*` e ignora todos os ficheiros novos. **9 fotografias foram renomeadas** em `images/` (com `git mv`; a lista de nomes antigos e novos está em `images/RENAMED_FILES.csv`, e os nomes originais continuam no branch `main` e no histórico do Git): `Solar1_10`, `Solqr2_10` e `Solar3_10.04.26` → `imitation1.1–1.3.10.04.26` (ImT.1); `boiled1–3.20.04.26` → `solar1–3.21.04.26` (Sol.3, 21/04); e gralhas: `immitation2.1` → `imitation2.1`, `SC2.SC1.05.05.26` → `SC2.05.05.26`, `SC3..05.05.26` → `SC3.05.05.26`. Os valores medidos não mudaram com a mudança de nome. A tabela manual `resultados_cor/color_by_object.csv` e os ficheiros antigos em `resultados_cor/` continuam com os nomes antigos (na tabela manual, as `boiled*.20.04` aparecem na coluna Boild, o que está errado). Ficaram de fora: `Slow*`, `Skim*`, `*WithoutSkin*`, `FastBNaked`, `FastBSkin`, `IMG_0444`, `solar_*_08.04.26` e os `.CR2`.
5. **Duplicados no script original:** no Windows, `main.py` analisa cada imagem duas vezes (`*.jpg` e `*.JPG` apanham os mesmos ficheiros), o que duplica o n. Este pipeline não tem esse problema. O `main.py` não foi alterado.
6. Na fotografia `Boiled3_23.04` há uma linha clara no bordo esquerdo que entra na máscara do grão; o efeito é desprezável (< 0,1% da área). A segmentação das restantes fotografias está correta (`grain_masks/`).
7. O `resultados_cor/color_by_object.csv` no repositório é uma tabela feita à mão e não resultado do script; seria apagado se o `main.py` voltasse a correr.
8. **O zoom variou entre sessões.** Segundo os EXIF, a distância focal das fotografias do grão vai de 24 a 49 mm (Canon EOS 60D; 12 valores diferentes) e a dos Pantone é 35 mm. A exposição (1/100 s, f/8, ISO 100) e o equilíbrio de brancos (manual) parecem fixos. Para a cor média, o zoom tem pouco efeito direto, mas pode mudar o vinhetamento e a resposta da lente. Além disso, as áreas em pixels (`area_pixels`) não são comparáveis entre sessões. Na Figura 5, cada fotografia foi reamostrada pelo fator f_ref/f para ficar com a mesma escala, o que assume que a câmara esteve sempre à mesma distância da amostra.
