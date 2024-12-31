#paper: https://onlinelibrary.wiley.com/doi/full/10.1002/adma.202304654
#Liposome: build from:
#DPPC: "CCCCCCCCCCCCCCCC(=O)OCC(COP(=O)([O-])OCC[N+](C)(C)C)OC(=O)CCCCCCCCCCCCCCC" (https://www.sigmaaldrich.com/IL/en/product/sigma/p4329?srsltid=AfmBOop3hx0ga6no51kkig1CelEIJMlP4HllTqoSV-m1PRkgRy0MpNyz)
#colesterol: "CC(C)CCCC(C)C1CCC2C3CC=C4CC(O)CCC4(C)C3CCC12C" (https://www.sigmaaldrich.com/IL/en/product/sigma/c8667?srsltid=AfmBOorNcjbvGlhk-hvd6TFoaC-bkTx6RN_6SNEGJqByyRh2GoX5LABK)
#DSPE-PEG(1000):"CCCCCCCCCCCCCCCCCC(=O)OCC(COP(=O)([O-])OCCNC(=O)CC=COOOOOOOOOOOOOOOOOOOOCCOOCC(=O)O)OC(=O)CCCCCCCCCCCCCCCCC" (https://www.sigmaaldrich.com/IL/en/product/avanti/880239p?srsltid=AfmBOoqcf_meeyh9jiFDjwsw_rez_hSp5g2-TLduZjb8tRcJGs59Gey6)
#DSPE-PEG(2000):"CCCCCCCCCCCCCCCCCC(=O)OCC(COP(=O)([O-])OCCNC(=O)OCCOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOC=CN)OC(=O)CCCCCCCCCCCCCCCCC" (https://www.sigmaaldrich.com/IL/en/product/avanti/880128p)
#
#all_liposome = DPPC + colesterol + DSPE-PEG(1000) + DSPE-PEG(2000)
#"CCCCCCCCCCCCCCCC(=O)OCC(COP(=O)([O-])OCC[N+](C)(C)C)OC(=O)CCCCCCCCCCCCCCC.CC(C)CCCC(C)C1CCC2C3CC=C4CC(O)CCC4(C)C3CCC12C.CCCCCCCCCCCCCCCCCC(=O)OCC(COP(=O)([O-])OCCNC(=O)CC=COOOOOOOOOOOOOOOOOOOOCCOOCC(=O)O)OC(=O)CCCCCCCCCCCCCCCCC.CCCCCCCCCCCCCCCCCC(=O)OCC(COP(=O)([O-])OCCNC(=O)OCCOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOC=CN)OC(=O)CCCCCCCCCCCCCCCCC"
#
#transferin:
#https://www.uniprot.org/uniprotkb/P02787/entry
#

#all_dirs:


python -m inference --config default_inference_args.yaml --protein_path '../transferin/transferrin.pdb' --ligand 'CCCCCCCCCCCCCCCC(=O)OCC(COP(=O)([O-])OCC[N+](C)(C)C)OC(=O)CCCCCCCCCCCCCCC' --out_dir '../transferin/DPPC'
python -m inference --config default_inference_args.yaml --protein_path '../transferin/transferrin.pdb' --ligand 'CC(C)CCCC(C)C1CCC2C3CC=C4CC(O)CCC4(C)C3CCC12C' --out_dir '../transferin/colesterol'
python -m inference --config default_inference_args.yaml --protein_path '../transferin/transferrin.pdb' --ligand 'CCCCCCCCCCCCCCCCCC(=O)OCC(COP(=O)([O-])OCCNC(=O)CC=COOOOOOOOOOOOOOOOOOOOCCOOCC(=O)O)OC(=O)CCCCCCCCCCCCCCCCC' --out_dir '../transferin/DSPE-PEG1000'
python -m inference --config default_inference_args.yaml --protein_path '../transferin/transferrin.pdb' --ligand 'CCCCCCCCCCCCCCCCCC(=O)OCC(COP(=O)([O-])OCCNC(=O)OCCOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOC=CN)OC(=O)CCCCCCCCCCCCCCCCC' --out_dir '../transferin/DSPE-PEG2000'
