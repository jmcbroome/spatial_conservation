rule all:
    input:
        "{target}_GRBs.bedgraph"

rule blat:
    input:
        "references/{target}.fa",
        "putative_CNEs.fa"
    output:
        "{target}_CNEblat.psl"
    shell:
        "./blat {input[0]} {input[1]} {output}"

rule process:
    input:
        "{target}_CNEblat.psl"
    output:
        "{target}_GRBs.bedgraph"
    shell:
        "python3 process_psl.py -p {input} -o {output}" #defaults of that script for now.
