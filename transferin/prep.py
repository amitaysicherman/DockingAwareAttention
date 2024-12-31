import numpy as np
import torch
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig
from preprocessing.get_esm3_fold_emb import ESM3FoldEmbedding
import os
seq = "MRLAVGALLVCAVLGLCLAVPDKTVRWCAVSEHEATKCQSFRDHMKSVIPSDGPSVACVKKASYLDCIRAIAANEADAVTLDAGLVYDAYLAPNNLKPVVAEFYGSKEDPQTFYYAVAVVKKDSGFQMNQLRGKKSCHTGLGRSAGWNIPIGLLYCDLPEPRKPLEKAVANFFSGSCAPCADGTDFPQLCQLCPGCGCSTLNQYFGYSGAFKCLKDGAGDVAFVKHSTIFENLANKADRDQYELLCLDNTRKPVDEYKDCHLAQVPSHTVVARSMGGKEDLIWELLNQAQEHFGKDKSKEFQLFSSPHGKDLLFKDSAHGFLKVPPRMDAKMYLGYEYVTAIRNLREGTCPEAPTDECKPVKWCALSHHERLKCDEWSVNSVGKIECVSAETTEDCIAKIMNGEADAMSLDGGFVYIAGKCGLVPVLAENYNKSDNCEDTPEAGYFAIAVVKKSASDLTWDNLKGKKSCHTAVGRTAGWNIPMGLLYNKINHCRFDEFFSEGCAPGSKKDSSLCKLCMGSGLNLCEPNNKEGYYGYTGAFRCLVEKGDVAFVKHQTVPQNTGGKNPDPWAKNLNEKDYELLCLDGTRKPVEEYANCHLARAPNHAVVTRKDKEACVHKILRQQQHLFGSNVTDCSGNFCLFRSETKDLLFRDDTVCLAKLHDRNTYEKYLGEEYVKAVGNLRKCSTSSLLEACTFRRP"
# get token from environment variable
token = os.environ.get("FOLD_TOKEN")
print(token)
model=ESM3FoldEmbedding(token)
fold, emb = model.get_fold_and_embedding(seq)
if emb is not None:
    print(emb.shape)
    np.save("transferin/emb_6B.npy", emb)
#
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# model = ESMC.from_pretrained("esmc_600m", device=device).eval()
# protein = ESMProtein(sequence=seq)
# protein = model.encode(protein)
# conf = LogitsConfig(return_embeddings=True, sequence=True)
# vec = model.logits(protein, conf).embeddings[0].detach().cpu().numpy()
# np.save("transferin/emb.npy", vec)
