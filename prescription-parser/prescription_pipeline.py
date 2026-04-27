"""
PrescriptionParserService
=========================

End-to-end pipeline for AI-powered handwritten prescription parsing.

Stages
------
1. Ingest the Egyptian medicines dataset (Kaggle) and clean it.
2. Run a Qwen2-VL vision-language model (HuggingFace transformers) on the
   prescription image to extract medications in strict JSON.
3. Normalize each VLM-extracted drug name against the cleaned Egyptian drugs
   list using RapidFuzz, attaching `confidence_score` and `official_match`.

Heavy components are lazy-loaded on first use. A live `status` dict reports
the current loading / inference stage so the chatbot can show progress.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

# Prevent transformers from importing TensorFlow (its bundled protobuf is
# incompatible with protobuf >= 4) and force pure-Python protobuf as a safety net.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

logger = logging.getLogger("prescription_pipeline")
if not logger.handlers:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

# Default to a small Qwen2-VL checkpoint that loads fast on a single GPU and
# avoids vLLM's ABI mismatch with newer torch builds. Override via env if you
# want a larger one (e.g. Qwen/Qwen2-VL-7B-Instruct).
DEFAULT_MODEL_ID = os.getenv("RX_VLM_MODEL", "Qwen/Qwen2-VL-2B-Instruct")
DEFAULT_KAGGLE_DATASET = os.getenv(
    "RX_KAGGLE_DATASET", "younaniskander/medicines-from-egyptian-pharmacies"
)
DEFAULT_FUZZ_THRESHOLD = float(os.getenv("RX_FUZZ_THRESHOLD", "82"))
DEFAULT_MAX_NEW_TOKENS = int(os.getenv("RX_MAX_NEW_TOKENS", "768"))


@dataclass
class ParsedPrescription:
    medications: List[Dict[str, Any]] = field(default_factory=list)
    raw_vlm_output: str = ""
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "medications": self.medications,
            "raw_vlm_output": self.raw_vlm_output,
            "notes": self.notes,
        }


SYSTEM_PROMPT = (
    "You are an expert pharmacist trained in reading handwritten medical "
    "prescriptions, including Arabic and English. Extract ONLY the medications "
    "from the image. Ignore patient personal information, doctor signatures, "
    "stamps, addresses, and free-text notes that are not medications.\n\n"
    "Output STRICTLY a single JSON array. Each element must be an object with "
    "exactly these keys: \"drug\" (string), \"dosage\" (string), \"frequency\" "
    "(string). Use empty string \"\" if a field is not legible.\n"
    "For frequency, copy the exact timing wording when present, especially Arabic "
    "timing such as: بعد الأكل / قبل الأكل / صباحا / مساء / كل 8 ساعات / قبل النوم.\n\n"
    "Critical rules:\n"
    "- DO NOT invent medications. If handwriting is unclear, output the closest "
    "  visible characters as best you can decode them.\n"
    "- DO NOT add markdown, explanations, or any text before or after the JSON.\n"
    "- DO NOT wrap the JSON in code fences.\n"
    "- If no medications are visible, return an empty JSON array: [].\n"
)

USER_PROMPT = (
    "Read this prescription image and extract every prescribed medication.\n"
    "Return only the JSON array as instructed."
)


class PrescriptionParserService:
    """Singleton-ish service: dataset + Qwen2-VL (transformers) + fuzzy match."""

    _instance: "Optional[PrescriptionParserService]" = None
    _lock = threading.Lock()

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        fuzz_threshold: float = DEFAULT_FUZZ_THRESHOLD,
        kaggle_dataset: str = DEFAULT_KAGGLE_DATASET,
    ) -> None:
        self.model_id = model_id
        self.fuzz_threshold = fuzz_threshold
        self.kaggle_dataset = kaggle_dataset

        self._model: Any = None
        self._processor: Any = None
        self._device: str = "cpu"
        self._dtype: Any = None
        self._drugs: Optional[List[str]] = None
        self._init_lock = threading.Lock()

        # Live progress for the chatbot to poll.
        self._status: Dict[str, Any] = {
            "stage": "idle",
            "message": "في وضع الاستعداد",
            "started_at": None,
            "updated_at": time.time(),
            "model_id": model_id,
            "model_loaded": False,
            "drugs_loaded": False,
        }

    def _set_status(self, stage: str, message: str) -> None:
        self._status["stage"] = stage
        self._status["message"] = message
        self._status["updated_at"] = time.time()
        if stage == "idle":
            self._status["started_at"] = None
        elif self._status.get("started_at") is None or stage == "done" or stage == "error":
            if self._status.get("started_at") is None:
                self._status["started_at"] = time.time()
        logger.info("[status] %s — %s", stage, message)

    def get_status(self) -> Dict[str, Any]:
        st = dict(self._status)
        if st.get("started_at"):
            st["elapsed_sec"] = round(time.time() - st["started_at"], 1)
        else:
            st["elapsed_sec"] = 0.0
        return st

    @classmethod
    def get_instance(cls) -> "PrescriptionParserService":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    # ------------------------------------------------------------------
    # Step 1: drug dataset
    # ------------------------------------------------------------------

    def _ensure_drugs_loaded(self) -> List[str]:
        if self._drugs is not None:
            return self._drugs
        with self._init_lock:
            if self._drugs is not None:
                return self._drugs
            self._set_status("loading_drugs",
                             "جاري تحميل قاعدة بيانات الأدوية المصرية...")
            self._drugs = self._load_egyptian_drugs()
            self._status["drugs_loaded"] = True
            self._set_status("drugs_ready",
                             f"تم تحميل {len(self._drugs)} اسم دواء.")
            return self._drugs

    def _load_egyptian_drugs(self) -> List[str]:
        import kagglehub  # type: ignore
        import pandas as pd  # type: ignore

        logger.info("Downloading Kaggle dataset: %s", self.kaggle_dataset)
        dataset_dir = Path(kagglehub.dataset_download(self.kaggle_dataset))
        csvs = sorted(dataset_dir.glob("**/*.csv"))
        if not csvs:
            raise FileNotFoundError(f"No CSV file found in {dataset_dir}")
        df = pd.read_csv(csvs[0])
        if "Drugname" not in df.columns:
            matches = [c for c in df.columns if c.lower() == "drugname"]
            if not matches:
                raise KeyError(
                    f"`Drugname` column not found. Available: {list(df.columns)}"
                )
            df = df.rename(columns={matches[0]: "Drugname"})
        df = df[["Drugname"]].copy()
        df["Drugname"] = df["Drugname"].dropna().astype(str).str.strip().str.lower()
        df = df[df["Drugname"].astype(bool)].drop_duplicates()
        return df["Drugname"].tolist()

    # ------------------------------------------------------------------
    # Step 2: Qwen2-VL via transformers
    # ------------------------------------------------------------------

    def _ensure_model_loaded(self) -> None:
        if self._model is not None:
            return
        with self._init_lock:
            if self._model is not None:
                return

            self._set_status(
                "loading_model",
                f"جاري تحميل نموذج الرؤية {self.model_id} (قد يستغرق وقتاً عند أول مرة)..."
            )

            import torch  # type: ignore
            from transformers import (  # type: ignore
                AutoProcessor,
                Qwen2VLForConditionalGeneration,
            )

            if torch.cuda.is_available():
                self._device = "cuda"
                self._dtype = torch.float16
            else:
                self._device = "cpu"
                self._dtype = torch.float32

            self._set_status(
                "loading_processor",
                "جاري تحميل المعالج (processor) للنموذج...",
            )
            self._processor = AutoProcessor.from_pretrained(
                self.model_id, trust_remote_code=True
            )

            self._set_status(
                "downloading_weights",
                "جاري تحميل أوزان النموذج من Hugging Face... (هيتحفظ كاش لمرات بعدين)",
            )
            self._model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_id,
                torch_dtype=self._dtype,
                device_map=self._device if self._device == "cuda" else None,
                trust_remote_code=True,
            )
            if self._device == "cpu":
                self._model = self._model.to(self._device)
            self._model.eval()

            self._status["model_loaded"] = True
            self._set_status("model_ready",
                             f"النموذج جاهز على {self._device.upper()}.")

    def _run_vlm(self, image: Any) -> str:
        self._ensure_model_loaded()
        import torch  # type: ignore

        self._set_status("inference",
                         "جاري قراءة الروشتة واستخراج الأدوية بالنموذج...")

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": USER_PROMPT},
                ],
            },
        ]
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt",
        ).to(self._device)

        with torch.inference_mode():
            generated_ids = self._model.generate(
                **inputs,
                max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
                do_sample=False,
            )
        trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        decoded = self._processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return (decoded[0] if decoded else "").strip()

    # ------------------------------------------------------------------
    # Step 3: parsing + fuzzy normalization
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_json_array(raw: str) -> List[Dict[str, Any]]:
        if not raw:
            return []
        try:
            value = json.loads(raw)
            if isinstance(value, list):
                return value
            if isinstance(value, dict):
                for k in ("medications", "drugs", "items"):
                    if isinstance(value.get(k), list):
                        return value[k]
                return [value]
        except json.JSONDecodeError:
            pass

        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(),
                         flags=re.MULTILINE)
        try:
            value = json.loads(cleaned)
            if isinstance(value, list):
                return value
        except json.JSONDecodeError:
            pass

        match = re.search(r"\[\s*(?:\{.*?\}\s*,?\s*)*\]", cleaned, re.DOTALL)
        if match:
            try:
                value = json.loads(match.group(0))
                if isinstance(value, list):
                    return value
            except json.JSONDecodeError:
                pass

        logger.warning("Could not parse VLM output as JSON. Raw: %r", raw[:300])
        return []

    @staticmethod
    def _normalize_text(value: Any) -> str:
        return re.sub(r"\s+", " ", str(value or "").strip())

    @staticmethod
    def _letters_only(value: str) -> str:
        return "".join(re.findall(r"[A-Za-z\u0621-\u064A]", str(value or "")))

    @classmethod
    def _humanize_schedule_ar(cls, raw_frequency: str, raw_dosage: str) -> Dict[str, str]:
        """Convert noisy frequency text into a clear Arabic timing line."""
        freq = cls._normalize_text(raw_frequency)
        dosage = cls._normalize_text(raw_dosage)
        low = freq.lower()

        if not low:
            return {
                "frequency": freq,
                "schedule_ar": "غير واضح من الروشتة",
                "schedule_source": "unknown",
            }

        source = "inferred"
        count_hint = ""
        parts: List[str] = []

        if re.search(r"\b(prn|عند\s*اللزو?م|عند\s*الحاجة)\b", low):
            return {
                "frequency": freq,
                "schedule_ar": "عند اللزوم",
                "schedule_source": "explicit",
            }

        every_hours = re.search(
            r"(?:\bq\s*)(\d{1,2})\s*h\b|(?:كل\s*)(\d{1,2})\s*(?:ساع|ساعة|ساعات)",
            low,
        )
        if every_hours:
            hours = every_hours.group(1) or every_hours.group(2)
            return {
                "frequency": freq,
                "schedule_ar": f"كل {hours} ساعات",
                "schedule_source": "explicit",
            }

        if re.search(r"\b(bid|b\.i\.d|مرتين|twice)\b", low):
            count_hint = "مرتين يوميا"
        elif re.search(r"\b(tid|t\.i\.d|ثلاث|3\s*مرات|three times)\b", low):
            count_hint = "3 مرات يوميا"
        elif re.search(r"\b(qid|q\.i\.d|اربع|4\s*مرات|four times)\b", low):
            count_hint = "4 مرات يوميا"
        elif re.search(r"\b(od|qd|once|daily|يومي|مرة)\b", low):
            count_hint = "مرة يوميا"

        if not count_hint:
            one_like = {"1", "1x", "1 tab", "1 cap", "1 amp", "1 drop", "1 dose"}
            two_like = {"2", "2x", "2 tab", "2 cap", "2 dose"}
            three_like = {"3", "3x", "3 tab", "3 cap", "3 dose"}
            if low in one_like:
                count_hint = "مرة يوميا"
            elif low in two_like:
                count_hint = "مرتين يوميا"
            elif low in three_like:
                count_hint = "3 مرات يوميا"

        if count_hint:
            parts.append(count_hint)

        if re.search(r"قبل\s*ال[اأ]?كل|before\s*meal|before\s*food", low):
            parts.append("قبل الأكل")
            source = "explicit"
        elif re.search(r"بعد\s*ال[اأ]?كل|after\s*meal|after\s*food", low):
            parts.append("بعد الأكل")
            source = "explicit"

        if re.search(r"قبل\s*النوم|before\s*sleep|bedtime|\bhs\b", low):
            parts.append("قبل النوم")
            source = "explicit"

        times = []
        if re.search(r"صباح|صبح|morning|\bam\b|الفطار", low):
            times.append("صباحا")
            source = "explicit"
        if re.search(r"ظهر|noon|الغدا", low):
            times.append("ظهرا")
            source = "explicit"
        if re.search(r"مساء|ليل|عشاء|bedtime|\bpm\b|\bhs\b|evening|night", low):
            times.append("مساء")
            source = "explicit"
        if times:
            parts.append(" و ".join(times))

        schedule_ar = " - ".join(parts).strip(" -")
        if not schedule_ar:
            schedule_ar = "غير واضح من الروشتة"

        # Many injections/vaccines are single-dose orders.
        if schedule_ar == "مرة يوميا":
            dlow = dosage.lower()
            if any(k in dlow for k in ("amp", "ampoule", "vial", "inj", "حقن")):
                schedule_ar = "جرعة واحدة"
                source = "inferred"

        if schedule_ar == "غير واضح من الروشتة":
            source = "unknown"

        return {
            "frequency": freq,
            "schedule_ar": schedule_ar,
            "schedule_source": source,
        }

    @classmethod
    def _is_plausible_match(cls, raw_drug: str, official_name: str) -> bool:
        """Reject obviously noisy fuzzy matches (e.g., 2-letter tokens)."""
        raw_letters = cls._letters_only(raw_drug).lower()
        off_letters = cls._letters_only(official_name).lower()

        if len(raw_letters) < 3:
            return False
        if len(off_letters) < 4 and len(raw_letters) >= 4:
            return False
        return True

    def _normalize_one(self, item: Dict[str, Any]) -> Dict[str, Any]:
        from rapidfuzz import process, fuzz  # type: ignore

        drugs = self._ensure_drugs_loaded()
        raw_drug = self._normalize_text(item.get("drug"))
        raw_dosage = self._normalize_text(item.get("dosage"))
        raw_frequency = self._normalize_text(item.get("frequency"))
        schedule_info = self._humanize_schedule_ar(raw_frequency, raw_dosage)
        normalized: Dict[str, Any] = {
            "drug_extracted": raw_drug,
            "drug": raw_drug,
            "dosage": raw_dosage,
            "frequency": schedule_info["frequency"],
            "schedule_ar": schedule_info["schedule_ar"],
            "schedule_source": schedule_info["schedule_source"],
            "confidence_score": 0.0,
            "official_match": False,
        }
        if not raw_drug or not drugs:
            return normalized
        if len(self._letters_only(raw_drug)) < 3:
            return normalized
        match = process.extractOne(
            raw_drug.lower(), drugs, scorer=fuzz.WRatio, score_cutoff=0
        )
        if match is None:
            return normalized
        official_name, score, _index = match
        normalized["confidence_score"] = round(float(score), 2)
        if score >= self.fuzz_threshold and self._is_plausible_match(raw_drug, official_name):
            normalized["drug"] = official_name
            normalized["official_match"] = True
        return normalized

    # ------------------------------------------------------------------
    # End-to-end
    # ------------------------------------------------------------------

    @staticmethod
    def _load_image(image_or_path: Any) -> Any:
        from PIL import Image  # type: ignore
        if hasattr(image_or_path, "convert"):
            img = image_or_path
        else:
            img = Image.open(str(image_or_path))
        return img.convert("RGB")

    def parse(self, image_or_path: Any) -> ParsedPrescription:
        # New request -> reset the elapsed counter.
        self._status["started_at"] = time.time()
        image = self._load_image(image_or_path)
        try:
            self._ensure_drugs_loaded()
            raw_output = self._run_vlm(image)
            self._set_status("post_processing",
                             "جاري المطابقة مع قاعدة الأدوية المصرية...")
            items = self._extract_json_array(raw_output)
            if not isinstance(items, list):
                items = []
            medications: List[Dict[str, Any]] = [
                self._normalize_one(it) for it in items if isinstance(it, dict)
            ]
            notes = (
                "Parsed and verified against Egyptian medicines dataset."
                if medications else "No medications extracted from the image."
            )
            self._set_status("done", "تم الانتهاء من تحليل الروشتة.")
            return ParsedPrescription(
                medications=medications,
                raw_vlm_output=raw_output,
                notes=notes,
            )
        except Exception as exc:
            self._set_status("error", f"حدث خطأ: {exc}")
            raise


def _main() -> None:
    parser = argparse.ArgumentParser(description="Parse a prescription image.")
    parser.add_argument("image")
    parser.add_argument("--threshold", type=float, default=DEFAULT_FUZZ_THRESHOLD)
    args = parser.parse_args()
    service = PrescriptionParserService(fuzz_threshold=args.threshold)
    result = service.parse(args.image)
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _main()
