# ARQ3-Otimiza-o-Paralela-de-GEMM

## GEMM — Multiplicação de Matrizes: CPU, OpenMP e CUDA

Implementação e análise de desempenho de multiplicação de matrizes quadradas (GEMM) com versões sequenciais em CPU, paralelização com OpenMP e aceleração em GPU via CUDA.

**Disciplina:** Arquitetura de Computadores — PUC Minas  
**Autores:** Davi Cândido de Almeida · Vitor Leite Setragni

---

## Visão geral

A operação implementada é `C = A × B` para matrizes quadradas de dimensão `N × N`. O desempenho é medido em GFLOP/s com base em `2N³` operações de ponto flutuante.

Foram implementadas 7 versões:

| Versão | Descrição |
|---|---|
| `naive` | Três laços aninhados, versão de referência |
| `transposed` | Transpõe B antes da multiplicação para melhorar localidade |
| `blocked` | Blocagem com tamanhos de bloco 16, 32 e 64 |
| `openmp` | Paralelização com OpenMP (1, 2, 4 e 8 threads) |
| `blocked_openmp` | Blocagem + OpenMP combinados (bloco 32) |
| `cuda_naive` | Kernel CUDA direto — cada thread calcula um elemento de C |
| `cuda_tiled` | Kernel CUDA com memória compartilhada em tiles 16×16 |

---

## Ambiente de execução

| Item | Configuração |
|---|---|
| CPU | Intel Xeon @ 2.00 GHz, x86_64, 2 CPUs lógicas |
| Caches | L1d 32 KiB · L1i 32 KiB · L2 1 MiB · L3 38,5 MiB |
| GPU | NVIDIA Tesla T4 — 15360 MiB |
| CUDA | Versão 13.0 |
| Compilador CPU | gcc com `-O3 -march=native -fopenmp` |
| Compilador GPU | nvcc |
| Repetições | 5 por configuração (tempo médio) |
| Validação | Erro absoluto máximo em relação à versão `naive` |

---

## Estrutura do projeto

```
.
├── gemm_cpu.c          # Versões CPU: naive, transposed, blocked, openmp, blocked_openmp
├── gemm_cuda.cu        # Versões GPU: cuda_naive, cuda_tiled
├── Trabalho_Pratico_3.ipynb  # Notebook com experimentos, gráficos e análise
└── README.md
```

---

## Como executar

### Pré-requisitos

- GCC com suporte a OpenMP
- NVCC (requer GPU NVIDIA — ativar GPU no Colab em *Ambiente de execução → Alterar tipo de ambiente de execução → GPU*)

### Compilação e execução CPU

```bash
gcc gemm_cpu.c -O3 -march=native -fopenmp -fopt-info-vec -o gemm_cpu -lm

# Uso: ./gemm_cpu <versão> <N> <repetições> [block_size]
./gemm_cpu naive       512 5
./gemm_cpu transposed  512 5
./gemm_cpu blocked     512 5 64
./gemm_cpu openmp      512 5
./gemm_cpu blocked_openmp 512 5 32
```

### Compilação e execução GPU

```bash
nvcc gemm_cuda.cu -O3 -o gemm_cuda

# Uso: ./gemm_cuda <naive|tiled> <N> <repetições>
./gemm_cuda naive 1024 5
./gemm_cuda tiled 1024 5
```

---

## Principais resultados

### CPU — melhores tempos por tamanho de matriz

| N | Referência (naive) | Melhor versão | Tempo médio | Speedup |
|---|---|---|---|---|
| 128 | 0,002742 s | blocked BS64 | 0,000493 s | 5,56× |
| 256 | 0,064802 s | blocked BS64 | 0,003200 s | 20,25× |
| 512 | 0,355613 s | blocked BS64 | 0,020699 s | 17,18× |

A blocagem com bloco 64 foi a abordagem mais eficiente em CPU, atingindo **12,97 GFLOP/s** para N=512 contra 0,75 GFLOP/s da versão ingênua.

### OpenMP — speedup e eficiência (N=512)

| Threads | Tempo médio | Speedup | Eficiência |
|---|---|---|---|
| 1 | 0,1806 s | 1,00× | 1,00 |
| 2 | 0,1279 s | 1,41× | 0,71 |
| 4 | 0,1246 s | 1,45× | 0,36 |
| 8 | 0,1762 s | 1,02× | 0,13 |

A eficiência cai com mais threads porque o ambiente tem apenas 2 processadores lógicos. Com 4 ou 8 threads há sobrecarga de escalonamento sem ganho físico proporcional.

### CUDA — kernel vs. tempo total

| N | Versão | Kernel (ms) | Total (ms) | Kernel GFLOP/s |
|---|---|---|---|---|
| 512 | cuda_naive | 0,487 | 1,562 | 551,07 |
| 512 | cuda_tiled | 0,321 | 1,371 | 835,65 |
| 1024 | cuda_naive | 5,789 | 9,204 | 370,94 |
| 1024 | cuda_tiled | 3,734 | 7,483 | 575,04 |

### GPU vs. CPU (tempo total, mesmos tamanhos)

| N | Melhor CPU | Melhor CUDA total | Speedup GPU/CPU |
|---|---|---|---|
| 128 | 0,493 ms | 0,202 ms | 2,44× |
| 256 | 3,200 ms | 0,361 ms | 8,87× |
| 512 | 20,699 ms | 1,371 ms | 15,10× |

O ganho da GPU aumenta com o tamanho da matriz, pois o custo computacional O(N³) cresce mais rápido que o custo de transferência O(N²).

---

## Conclusões

- A **blocagem** foi a otimização mais impactante em CPU por melhorar o reúso de dados em cache.
- O **OpenMP** não escalou linearmente no ambiente testado (2 núcleos físicos); configurações com 4 e 8 threads geraram overhead sem ganho proporcional.
- O **cuda_tiled** superou o cuda_naive em todos os tamanhos testados ao reduzir acessos à memória global com memória compartilhada.
- O tempo total de aplicação GPU inclui cópias CPU↔GPU — esse custo deve ser considerado ao comparar com CPU.
- Todas as versões foram validadas com erro máximo inferior a 0,0001 em relação à versão de referência.

---

## Referências

- NETLIB. [BLAS: Basic Linear Algebra Subprograms](https://www.netlib.org/blas/)
- NVIDIA. [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- OpenMP ARB. [OpenMP API Specification v5.2](https://www.openmp.org/specifications/)
