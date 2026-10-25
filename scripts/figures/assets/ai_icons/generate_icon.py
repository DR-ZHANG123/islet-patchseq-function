"""Generate one explanatory icon with an image-generation API (endpoint and key read from IMAGE_API_BASE / IMAGE_API_KEY; never stored)."""
import os, sys, json, base64, urllib.request
STYLE = ("Flat vector-style scientific illustration icon in the style of BioRender, clean smooth shapes, soft subtle "
         "shading, thin dark outlines, isolated single object centred on a fully transparent background, no text, no "
         "letters, no labels, no shadow on the ground, colour palette: teal #2A9D8F, light teal #CDEBE6, warm orange "
         "#E8A33D, soft purple #9B7FC1, muted red #C0392B, steel blue #2E6DB4, light grey #E6E8EB.")
def gen(name, subject, size="1024x1024", model="gpt-image-2"):
    body = {"model": model, "prompt": f"{subject} {STYLE}", "size": size, "n": 1, "background": "transparent",
            "output_format": "png", "quality": "high"}
    req = urllib.request.Request(os.environ["IMAGE_API_BASE"].rstrip("/") + "/images/generations",
                                 data=json.dumps(body).encode(), method="POST",
                                 headers={"Authorization": "Bearer " + os.environ["IMAGE_API_KEY"],
                                          "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.load(r)
    item = d["data"][0]
    if item.get("b64_json"):
        png = base64.b64decode(item["b64_json"])
    else:
        png = urllib.request.urlopen(item["url"], timeout=300).read()
    open(f"{name}.png", "wb").write(png)
    print(name, len(png), "bytes")
if __name__ == "__main__":
    gen(sys.argv[1], sys.argv[2])
