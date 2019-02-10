from pyfaidx import Fasta
import logging
logging.basicConfig(level=logging.INFO)
from cyvcf2 import VCF

k = 31
max_deletion = 10

variants = "1000genomes_variants.vcf.gz"
ref = str(Fasta("linear_ref.fa")["6"])
vcf = VCF(variants)

for pos in range(0, len(ref)):

    n_variants = 0
    for v in vcf('6:%d-%d' % (pos, pos + k)):
        n_variants += 1
    print("POS: %d: %d" % (pos, n_variants))




