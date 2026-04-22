import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

function findWidget(node, name) {
  return (node.widgets || []).find((w) => w.name === name);
}

function appendUniqueLines(oldText, lines) {
  const existed = new Set((oldText || "").split(/\r?\n/).map((x) => x.trim()).filter(Boolean));
  for (const line of lines) {
    const t = (line || "").trim();
    if (!t) continue;
    existed.add(t);
  }
  return Array.from(existed).sort((a, b) => a.localeCompare(b, "zh-Hans-CN")).join("\n");
}

function removeLegacyInputs(node) {
  if (!node?.inputs?.length) return;
  const legacy = new Set(["custom_model_name", "custom_api_url", "image", "image_2", "image_3", "image_4"]);
  for (let i = node.inputs.length - 1; i >= 0; i -= 1) {
    const input = node.inputs[i];
    if (input && legacy.has(input.name)) {
      node.removeInput(i);
    }
  }
}

app.registerExtension({
  name: "gpt5.kb.upload",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== "GPT5ChatNode") return;

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onNodeCreated?.apply(this, arguments);
      removeLegacyInputs(this);

      const kbWidget = findWidget(this, "knowledge_files");
      if (!kbWidget) return r;

      const input = document.createElement("input");
      input.type = "file";
      input.multiple = true;
      input.accept = ".md,text/markdown";
      input.style.display = "none";
      document.body.appendChild(input);

      this.addWidget("button", "Upload .md Files", null, async () => {
        input.onchange = async () => {
          const files = Array.from(input.files || []);
          if (!files.length) return;

          const formData = new FormData();
          for (const f of files) formData.append("files", f, f.name);

          try {
            const resp = await api.fetchApi("/gpt5/upload_md", {
              method: "POST",
              body: formData,
            });
            const data = await resp.json();
            const names = (data.files || []).map((x) => x.name).filter(Boolean);
            kbWidget.value = appendUniqueLines(kbWidget.value, names);
            this.setDirtyCanvas(true, true);
          } catch (e) {
            console.error("[GPT5] upload md failed", e);
          } finally {
            input.value = "";
          }
        };
        input.click();
      });

      this.addWidget("button", "Clear Knowledge Files", null, () => {
        kbWidget.value = "";
        this.setDirtyCanvas(true, true);
      });

      return r;
    };

    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      const r = onConfigure?.apply(this, arguments);
      removeLegacyInputs(this);
      return r;
    };
  },
});
