"""Build Kaggle versions of the workshop notebooks, for when Colab runs out of GPU.

Run from the repo root after changing a notebook in notebooks/ (and after make_solutions.py):

    python3 scripts/make_kaggle_notebooks.py

Writes notebooks/kaggle/0X_..._kaggle.ipynb, and solutions/kaggle/0X_..._solutions_kaggle.ipynb
if the solutions copies exist. What changes for Kaggle:
- GPU and internet setup text (Kaggle's settings panel instead of Colab's Runtime menu),
- only one of Kaggle's two T4s is used, so results match Colab,
- no Google Drive: work is zipped and downloaded instead, and Notebook 3 finds an adapter
  the learner uploaded under /kaggle/input,
- /content paths become /kaggle/working.

If a source notebook changes so that a replacement no longer matches, the script stops and names
the cell, so the Kaggle copies can't silently go out of date.
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

BANNER = ("> **Kaggle version.** The same notebook as the Colab version, changed to run on Kaggle: "
          "the GPU settings, and you **download** your work instead of saving it to Google Drive.\n\n")

GPU_SETUP = """### Before you start: turn on the GPU and the internet
In the panel on the right of the Kaggle editor, under **Session options** (or the **Settings** menu):
- **Accelerator:** choose **GPU T4 x2**, the same GPU as Colab. Don't choose P100.
- **Internet:** turn it **on**, so the notebook can install packages and download the model.

Both need a phone-verified Kaggle account (**Settings → Phone verification** on kaggle.com). Free Kaggle gives about 30 GPU hours a week. We use only one of the two GPUs, so results match the Colab version."""

ONE_GPU = ('import os\n'
           'os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Kaggle has two T4s; use one, as on Colab (set before importing torch)\n\n')

KAGGLE_META = {"accelerator": "nvidiaTeslaT4", "isGpuEnabled": True,
               "isInternetEnabled": True, "dataSources": [], "language": "python",
               "sourceType": "notebook"}

# Any of these left in a Kaggle copy is a sign something was missed.
LEFTOVER = re.compile(r"colab|google drive|\bdrive\b|DRIVE_DIR|/content|Runtime →", re.I)
ALLOWED = ["Colab version", "as on Colab", "same GPU as Colab", "match the Colab", "free Colab",
           "Colab Pro", "Colab and Kaggle versions", "instead of saving it to Google Drive",
           "Kaggle has no Google Drive"]


def lines(text):
    parts = text.split("\n")
    return [p + "\n" for p in parts[:-1]] + ([parts[-1]] if parts[-1] else [])


class Notebook:
    def __init__(self, src, out):
        self.src, self.out = src, out
        self.nb = json.loads(src.read_text())
        cells = self.nb["cells"]
        cells[0]["source"] = lines(BANNER + "".join(cells[0]["source"]))
        meta = self.nb.setdefault("metadata", {})
        meta.pop("colab", None)
        meta["kaggle"] = KAGGLE_META

    def find(self, snippet):
        hits = [i for i, c in enumerate(self.nb["cells"]) if snippet in "".join(c["source"])]
        assert len(hits) == 1, f"{self.src.name}: {len(hits)} cells contain {snippet[:60]!r}"
        return hits[0]

    def text(self, i):
        return "".join(self.nb["cells"][i]["source"])

    def replace(self, old, new):
        i = self.find(old)
        s = self.text(i)
        assert s.count(old) == 1, f"{self.src.name}: {old[:60]!r} appears more than once in its cell"
        self.nb["cells"][i]["source"] = lines(s.replace(old, new))

    def set_cell(self, snippet, new):
        self.nb["cells"][self.find(snippet)]["source"] = lines(new)

    def prepend(self, snippet, text):
        i = self.find(snippet)
        self.nb["cells"][i]["source"] = lines(text + self.text(i))

    def replace_everywhere(self, old, new):
        for c in self.nb["cells"]:
            s = "".join(c["source"])
            if old in s:
                c["source"] = lines(s.replace(old, new))

    def save(self):
        for i, c in enumerate(self.nb["cells"]):
            s = "".join(c["source"])
            for a in ALLOWED:
                s = s.replace(a, "")
            m = LEFTOVER.search(s)
            assert not m, f"{self.out.name}: cell {i} still mentions {m.group(0)!r}"
        self.out.parent.mkdir(parents=True, exist_ok=True)
        self.out.write_text(json.dumps(self.nb, indent=1, ensure_ascii=False) + "\n")
        print("wrote", self.out.relative_to(ROOT))


def common(nb):
    nb.replace_everywhere("**GPU:** free Colab T4", "**GPU:** free Kaggle T4")
    nb.set_cell("Runtime → Change runtime type", GPU_SETUP)
    nb.replace_everywhere(" (double-click to see the code)", "")


def download_cell_md(name, why):
    return f"""### Download your adapter

Kaggle has no Google Drive, and it deletes files in `/kaggle/working` when your session ends. So zip the adapter and **download it to your computer**: click the link the next cell prints, or find `{name}.zip` under **Output** in the right-hand panel. {why}"""


def download_cell_code(name, folder):
    return f"""import shutil
from IPython.display import FileLink

shutil.make_archive("{name}", "zip", base_dir="{folder}")  # writes {name}.zip
FileLink("{name}.zip")  # click to download"""


def pirate(nb):
    common(nb)
    nb.replace("*The code in the next cell is hidden, because you don't need to read it. Run it.*",
               "*You don't need to read the code in the next cell. Run it.*")
    nb.replace("restart the session (**Runtime → Restart session**)",
               "restart the kernel (**Run → Restart & clear cell outputs**)")
    nb.prepend('MODEL_NAME = "unsloth/Qwen3.5-0.8B"\n\nmodel, tokenizer = FastLanguageModel.from_pretrained(', ONE_GPU)
    nb.set_cell("### Save to Google Drive", download_cell_md(
        "pirate_lora", "Notebook 3 shows how to upload it again."))
    nb.set_cell('shutil.copytree("pirate_lora"', download_cell_code("pirate_lora", "pirate_lora"))


def genomics(nb):
    common(nb)
    nb.replace("Colab already has most of what we need.", "Kaggle already has most of what we need.")
    nb.replace("removes Colab's preinstalled `torchao`: it isn't used here, and its old version makes recent PEFT fail.",
               "removes the preinstalled `torchao`, if there is one: it isn't used here, and old versions make recent PEFT fail.")
    nb.replace("# Colab's old torchao breaks recent PEFT;", "# old torchao versions break recent PEFT;")
    nb.prepend('MODEL_NAME = "InstaDeepAI/nucleotide-transformer-500m-human-ref"', ONE_GPU)
    nb.set_cell("### Save to Google Drive", download_cell_md(
        "genomics_lora", "Keep it if you want to reuse the classifier later."))
    nb.set_cell('shutil.copytree("promoter_lora"', download_cell_code("genomics_lora", "promoter_lora"))


def taking_it_home(nb):
    common(nb)
    nb.replace("Reload your pirate adapter from Google Drive,",
               "Reload your pirate adapter (the zip you downloaded in Notebook 1),")
    nb.replace("*Hidden code: run it.*", "*You don't need to read the code in the next cell. Run it.*")

    nb.set_cell("Connect Google Drive, so this notebook", """Set up the folder this notebook saves into, and use one of Kaggle's two GPUs.

Kaggle has no Google Drive, and it deletes `/kaggle/working` when your session ends. So everything you make here goes in one folder, `/kaggle/working/fine-tuning-workshop`, and you **download** what you want to keep. The cells print download links as you go; the files are also under **Output** in the right-hand panel.""")
    nb.set_cell('from google.colab import drive', """import os
from IPython.display import FileLink, display

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Kaggle has two T4s; use one, as on Colab (set before importing torch)

SAVE_DIR = "/kaggle/working/fine-tuning-workshop"
os.makedirs(SAVE_DIR, exist_ok=True)  # creates the folder if it isn't there yet

def download_link(path):
    display(FileLink(os.path.relpath(path)))  # prints a link that downloads the file""")
    nb.replace_everywhere("DRIVE_DIR", "SAVE_DIR")

    # 1a: find an uploaded adapter instead of reading Drive
    nb.replace("Colab deletes everything when the session ends, which is why section 7 of Notebook 1 saved your adapter to Drive.",
               "Kaggle deletes everything when the session ends, which is why section 7 of Notebook 1 had you download your adapter as `pirate_lora.zip`.")
    nb.replace("If your adapter isn't in Drive (for example, your training didn't finish), the cell downloads",
               """**To use your own adapter,** upload it before running the next cell: in the right-hand panel, click **Add Input → Upload**, choose `pirate_lora.zip`, give the dataset a name, and click **Create**. It's private unless you share it. Kaggle puts it under `/kaggle/input`, and the cell finds it there.

If the cell finds no uploaded adapter (for example, your training didn't finish), it downloads""")
    nb.set_cell("PIRATE_LORA_DIR = ", """import glob
import json

FALLBACK_URL = "https://raw.githubusercontent.com/UoA-eResearch/fine-tuning_workshop/main/adapters/pirate_lora.zip"   # the adapter we trained before the workshop

# In case Kaggle kept your upload as a zip, unzip it first
for z in glob.glob("/kaggle/input/**/pirate_lora*.zip", recursive=True):
    !unzip -q -o {z} -d /kaggle/working/uploaded

# Any uploaded folder holding a LoRA adapter for our base model
found = [os.path.dirname(p)
         for p in glob.glob("/kaggle/input/**/adapter_config.json", recursive=True)
                + glob.glob("/kaggle/working/uploaded/**/adapter_config.json", recursive=True)
         if json.load(open(p)).get("base_model_name_or_path") == MODEL_NAME]

if found:
    adapter_dir = found[0]
    print("Using your uploaded adapter:", adapter_dir)
else:
    adapter_dir = "/kaggle/working/pirate_lora_workshop"
    !wget -q {FALLBACK_URL} -O pirate_lora.zip  # downloads the zip from GitHub
    !unzip -q -o -j pirate_lora.zip -d {adapter_dir}  # -j drops the folders inside the zip
    print("No uploaded adapter found, so using the workshop's copy")

model, tokenizer = FastLanguageModel.from_pretrained(
    adapter_dir,  # loads the base model and attaches the adapter
    max_seq_length=1024,
    load_in_4bit=False,
)
FastLanguageModel.for_inference(model)  # faster generation mode""")

    # 1b: GGUF goes to the save folder, with download links
    nb.replace("Copy the GGUF file to Drive, and write a **Modelfile** next to it.",
               "Move the GGUF file into the save folder, and write a **Modelfile** next to it. Then download both.")
    nb.replace("shutil.copy(GGUF_NAME, f\"{GGUF_DIR}/{GGUF_NAME}\")",
               "shutil.move(GGUF_NAME, f\"{GGUF_DIR}/{GGUF_NAME}\")  # move, not copy: it's 0.8 GB")
    nb.replace("!ls -lh {GGUF_DIR}  # -h shows sizes in MB/GB",
               "!ls -lh {GGUF_DIR}  # -h shows sizes in MB/GB\n"
               "download_link(f\"{GGUF_DIR}/{GGUF_NAME}\")  # about 0.8 GB, so it takes a few minutes\n"
               "download_link(f\"{GGUF_DIR}/Modelfile\")")
    nb.replace("2. In Google Drive, download the `fine-tuning-workshop/pirate_gguf` folder (the `.gguf` file and the `Modelfile`), and unzip it if Drive zips it. Keep both files in the same folder.",
               "2. Download the `.gguf` file and the `Modelfile` with the links in 1b (or from **Output** in the right-hand panel, in `fine-tuning-workshop/pirate_gguf`). Put both files in the same folder, and check the browser didn't rename `Modelfile` (for example to `Modelfile.txt`).")

    # 2a: own CSV comes from an uploaded Kaggle dataset
    nb.replace('either a URL or a path in your Drive (for example `f"{SAVE_DIR}/my_data.csv"`; upload the file to the `fine-tuning-workshop` folder first).',
               "either a URL or a file you uploaded to Kaggle (**Add Input → Upload**, then copy the file's path from the **Input** panel; it starts with `/kaggle/input/`). Uploaded datasets are private, but they're stored by Kaggle, so check your data is allowed to go there.")
    nb.replace("# a URL, or a file path in Drive", "# a URL, or a path under /kaggle/input")

    # 2h, 2i: save, zip and link
    nb.replace("The results are saved to Drive, ready to open in Excel, R or pandas.",
               "The results are saved to a CSV with a download link, ready to open in Excel, R or pandas.")
    nb.replace('print(f"Saved {len(new_df)} predictions to {OUT_DIR}/predictions.csv")',
               'print(f"Saved {len(new_df)} predictions to {OUT_DIR}/predictions.csv")\n'
               'download_link(f"{OUT_DIR}/predictions.csv")')
    nb.replace("Saved to Drive, next to your predictions. To use it later, load it as in section 1a.",
               "Saved next to your predictions and zipped, ready to download. To use it later, upload the zip and load it as in section 1a.")
    nb.replace("!ls -lh {OUT_DIR}/lora",
               "!ls -lh {OUT_DIR}/lora\n\n"
               "import shutil\n"
               'shutil.make_archive(f"{OUT_DIR}/lora", "zip", root_dir=OUT_DIR, base_dir="lora")  # writes lora.zip\n'
               'download_link(f"{OUT_DIR}/lora.zip")')

    # 4 and recap
    nb.replace("- **Off campus:** free Colab has daily GPU limits; Colab Pro and Kaggle notebooks give more GPU time.",
               "- **Off campus:** free Kaggle gives about 30 GPU hours a week; free Colab has daily limits, and Colab Pro gives more.")
    nb.replace("(many models and tasks, ready for Colab)", "(many models and tasks, with Colab and Kaggle versions)")
    nb.replace("save adapters to Drive;", "download your adapters;")


PORTS = {"01_pirate_unsloth": pirate,
         "02_genomics_transformers": genomics,
         "03_taking_it_home": taking_it_home}

for stem, port in PORTS.items():
    jobs = [(ROOT / "notebooks" / f"{stem}.ipynb", ROOT / "notebooks" / "kaggle" / f"{stem}_kaggle.ipynb"),
            (ROOT / "solutions" / f"{stem}_solutions.ipynb", ROOT / "solutions" / "kaggle" / f"{stem}_solutions_kaggle.ipynb")]
    for src, out in jobs:
        if src.exists():
            nb = Notebook(src, out)
            port(nb)
            nb.save()
