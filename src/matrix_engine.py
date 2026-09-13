import csv
import logging
import math
import os
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Resolved against the project root so the engine works no matter which
# directory uvicorn was started from.
DEFAULT_CSV_PATH = os.path.join(_PROJECT_ROOT, "data", "symptom_disease_matrix.csv")


class MatrixEngine:
    """Symptom-disease matrix engine."""

    def __init__(self, csv_path: str = DEFAULT_CSV_PATH):
        self.csv_path = csv_path
        self.matrix_data: Dict[str, Any] = {}
        self._load_matrix()

    # Column groups in the flat CSV
    BREED_COLS = ["BR01", "BR02", "BR03", "BR04"]
    GENDER_COLS = ["M", "F"]
    WEIGHT_COLS = ["W01", "W02", "W03"]
    AGE_COLS = ["A01", "A02", "A03"]

    def _load_matrix(self) -> None:
        """Load matrix from flat CSV.

        Expected CSV layout:
            DISEASES, BR01..BR04, M, F, W01..W03, A01..A03, SY001..SYnnn
        - Breed/gender columns: binary flags (1 = applicable, 0 = not applicable).
        - Weight/age/symptom columns: integer weight (1-3, or negative); 0/blank
          means no association and is not stored in the in-memory matrix.
        """
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(
                f"Matrix file not found: {self.csv_path}"
            )

        try:
            with open(self.csv_path, "r", encoding="utf-8-sig", newline="") as file:
                reader = csv.DictReader(file)

                # Strip whitespace from header names (e.g. "M " -> "M")
                reader.fieldnames = [
                    name.strip() for name in reader.fieldnames
                ]

                symptom_cols = [
                    col for col in reader.fieldnames
                    if col.startswith("SY")
                ]

                self.matrix_data = {
                    "metadata": {
                        "breeds": list(self.BREED_COLS),
                        "genders": list(self.GENDER_COLS),
                        "weights": list(self.WEIGHT_COLS),
                        "ages": list(self.AGE_COLS),
                        "symptoms": symptom_cols,
                    },
                    "diseases": {},
                }

                for row in reader:
                    # Normalise keys/values
                    row = {
                        k.strip(): (v.strip() if v else "")
                        for k, v in row.items()
                    }

                    disease_id = row.get("DISEASES", "").strip()
                    if not disease_id:
                        continue

                    # Breeds / genders: binary flag columns (1 → included)
                    breeds = [
                        col for col in self.BREED_COLS
                        if self._parse_binary_flag(row.get(col, ""))
                    ]
                    genders = [
                        col for col in self.GENDER_COLS
                        if self._parse_binary_flag(row.get(col, ""))
                    ]

                    # Weights / ages / symptoms: parse integer values
                    weights = self._parse_int_cols(row, self.WEIGHT_COLS)
                    ages = self._parse_int_cols(row, self.AGE_COLS)
                    symptoms = self._parse_int_cols(row, symptom_cols)

                    self.matrix_data["diseases"][disease_id] = {
                        "breeds": breeds,
                        "genders": genders,
                        "weights": weights,
                        "ages": ages,
                        "symptoms": symptoms,
                    }

            logger.info("Loaded matrix from %s", self.csv_path)

        except Exception as exc:
            raise RuntimeError(
                f"Failed to load matrix: {exc}"
            ) from exc

    @staticmethod
    def _parse_binary_flag(val: Optional[str]) -> bool:
        """Parse a categorical 0/1 flag column (breed/gender).

        '1' → True, '0' or blank → False. Falls back to the legacy 'X'
        marker for backward compatibility with older matrix files.
        """
        val = (val or "").strip()
        if not val:
            return False
        try:
            return int(val) != 0
        except ValueError:
            return val.upper() == "X"

    @staticmethod
    def _parse_int_cols(
        row: Dict[str, str], cols: List[str],
    ) -> Dict[str, int]:
        """Extract non-zero integer values from *cols* in *row*.

        Blank cells and explicit '0' both mean "no association" and are
        omitted from the result, keeping the in-memory matrix sparse even
        though the CSV on disk is fully zero-filled.
        """
        result: Dict[str, int] = {}
        for col in cols:
            val = row.get(col, "").strip()
            if not val:
                continue
            try:
                parsed = int(val)
            except ValueError:
                continue
            if parsed != 0:
                result[col] = parsed
        return result

    def _save_matrix(self) -> None:
        """Save matrix back to flat CSV (same layout as the source file)."""
        directory = os.path.dirname(os.path.abspath(self.csv_path))
        os.makedirs(directory, exist_ok=True)

        symptom_cols = self.matrix_data.get(
            "metadata", {},
        ).get("symptoms", [])

        fields = (
            ["DISEASES"]
            + self.BREED_COLS
            + self.GENDER_COLS
            + self.WEIGHT_COLS
            + self.AGE_COLS
            + symptom_cols
        )

        with open(
            self.csv_path,
            "w",
            encoding="utf-8-sig",
            newline="",
        ) as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()

            for disease_id, disease in self.matrix_data.get(
                "diseases", {},
            ).items():
                row: Dict[str, str] = {"DISEASES": disease_id}

                # Breeds / genders → binary 0/1 flags
                for col in self.BREED_COLS:
                    row[col] = (
                        "1" if col in disease.get("breeds", []) else "0"
                    )
                for col in self.GENDER_COLS:
                    row[col] = (
                        "1" if col in disease.get("genders", []) else "0"
                    )

                # Weights / ages / symptoms → integer strings, zero-filled
                for col in self.WEIGHT_COLS:
                    val = disease.get("weights", {}).get(col, 0)
                    row[col] = str(val)
                for col in self.AGE_COLS:
                    val = disease.get("ages", {}).get(col, 0)
                    row[col] = str(val)
                for col in symptom_cols:
                    val = disease.get("symptoms", {}).get(col, 0)
                    row[col] = str(val)

                writer.writerow(row)

    #: Accepted spellings for each breed column, in CSV column order.
    BREED_ALIASES = {
        "BR01": {"BR001", "BR01", "DOG", "CHO", "CHÓ"},
        "BR02": {"BR002", "BR02", "CAT", "MEO", "MÈO"},
        "BR03": {"BR003", "BR03", "RABBIT", "THO", "THỎ"},
        "BR04": {"BR004", "BR04", "HAMSTER", "CHUOT HAMSTER"},
    }

    def _normalize_breed(self, breed: Optional[str]) -> str:
        if not breed:
            return "BR01"

        breed = breed.strip().upper()

        for column, aliases in self.BREED_ALIASES.items():
            if breed in aliases:
                return column

        return breed

    def _normalize_gender(self, gender: Optional[str]) -> str:
        if not gender:
            return "M"

        return "F" if gender.strip().upper() in {"F", "FEMALE"} else "M"

    def predict(
        self,
        pet_info: Dict[str, Any],
        symptoms: List[Dict[str, Any]],
        top_k: int = 5,
        min_rate: float = 0.10,
    ) -> List[Dict[str, Any]]:

        diseases = self.matrix_data.get("diseases", {})

        if not diseases:
            return []

        breed = self._normalize_breed(pet_info.get("breed"))
        gender = self._normalize_gender(pet_info.get("gender"))
        weight = str(pet_info.get("weight", "")).strip().upper()
        age = str(pet_info.get("age", "")).strip().upper()

        symptom_map: Dict[str, int] = {}
        for item in symptoms:
            code = item.get("symptom")
            if code:
                intensity = int(item.get("intensity", 1))
                symptom_map[code] = max(
                    symptom_map.get(code, 1), intensity,
                )

        if not symptom_map:
            return []

        total_intensity = sum(symptom_map.values())
        norm_pet = math.sqrt(
            sum(value ** 2 for value in symptom_map.values())
        )

        scores = []

        for disease_id, disease in diseases.items():
            breeds = disease.get("breeds", [])
            genders = disease.get("genders", [])

            if breeds and breed not in breeds:
                continue

            if genders and gender not in genders:
                continue

            disease_symptoms = disease.get("symptoms", {})

            matched = {
                code: symptom_map[code]
                for code in symptom_map
                if code in disease_symptoms
            }

            if not matched:
                continue

            dot_product = sum(
                disease_symptoms[code] * symptom_map[code]
                for code in matched
            )

            coverage = len(matched) / len(symptom_map)

            norm_disease = math.sqrt(
                sum(weight ** 2 for weight in disease_symptoms.values())
            )

            cosine = (
                dot_product / (norm_disease * norm_pet)
                if norm_disease * norm_pet
                else 0
            )

            demographic_bonus = 0

            if age in disease.get("ages", {}):
                demographic_bonus += disease["ages"][age] * 0.05

            if weight in disease.get("weights", {}):
                demographic_bonus += disease["weights"][weight] * 0.05

            normalized_dot = dot_product / (total_intensity * 3)

            score = (
                0.45 * normalized_dot
                + 0.35 * coverage
                + 0.20 * cosine
                + demographic_bonus
            )

            rate = 1 / (1 + math.exp(-6 * (score - 0.45)))
            rate = round(min(0.98, max(0.05, rate)), 2)

            if rate >= min_rate:
                scores.append(
                    {
                        "disease": disease_id,
                        "prevalence_rate": rate,
                        "_score": score,
                    }
                )

        scores.sort(
            key=lambda item: (
                item["prevalence_rate"],
                item["_score"],
            ),
            reverse=True,
        )

        return [
            {
                "disease": item["disease"],
                "prevalence_rate": item["prevalence_rate"],
            }
            for item in scores[:top_k]
        ]

    def readjust_weights(
        self,
        target_disease_id: str,
        symptoms: List[Dict[str, Any]],
        pet_info: Optional[Dict[str, Any]] = None,
        learning_rate: int = 1,
        save: bool = True,
    ) -> Dict[str, Any]:

        disease_id = target_disease_id.strip()
        diseases = self.matrix_data.setdefault("diseases", {})

        disease = diseases.setdefault(
            disease_id,
            {
                "breeds": list(self.BREED_COLS),
                "genders": list(self.GENDER_COLS),
                "weights": {},
                "ages": {},
                "symptoms": {},
            },
        )

        changes = {
            "disease_id": disease_id,
            "updated_symptoms": {},
        }

        for item in symptoms:
            symptom = item.get("symptom")

            if not symptom:
                continue

            old_weight = disease["symptoms"].get(symptom, 0)
            new_weight = min(5, old_weight + learning_rate)

            disease["symptoms"][symptom] = new_weight

            changes["updated_symptoms"][symptom] = {
                "old": old_weight,
                "new": new_weight,
            }

        if pet_info:
            breed = self._normalize_breed(pet_info.get("breed"))
            gender = self._normalize_gender(pet_info.get("gender"))

            if breed not in disease["breeds"]:
                disease["breeds"].append(breed)

            if gender not in disease["genders"]:
                disease["genders"].append(gender)

        if save:
            self._save_matrix()

        logger.info(
            "Updated matrix for %s: %s",
            disease_id,
            changes,
        )

        return changes

    def readjust_weights_batch(
        self,
        target_disease_ids: List[str],
        symptoms: List[Dict[str, Any]],
        pet_info: Optional[Dict[str, Any]] = None,
        learning_rate: int = 1,
    ) -> List[Dict[str, Any]]:
        """Learn multiple ground-truth diseases from the same case.

        Used when the caller passes a `diseases` array: every listed
        disease is reinforced with the same reported symptoms/pet-info,
        and the matrix CSV is written to disk only once at the end.
        """
        changes = [
            self.readjust_weights(
                target_disease_id=disease_id,
                symptoms=symptoms,
                pet_info=pet_info,
                learning_rate=learning_rate,
                save=False,
            )
            for disease_id in target_disease_ids
            if disease_id and disease_id.strip()
        ]

        self._save_matrix()

        return changes

    # ------------------------------------------------------------------
    # Editing API (used by the dashboard's matrix editor)
    # ------------------------------------------------------------------

    #: Editable range for weighted (weight / age / symptom) cells.
    CELL_MIN = -3
    CELL_MAX = 3

    @property
    def symptom_cols(self) -> List[str]:
        return list(
            self.matrix_data.get("metadata", {}).get("symptoms", [])
        )

    @staticmethod
    def _symptom_sort_key(code: str):
        match = re.match(r"^SY(\d+)$", code.strip().upper())
        return (0, int(match.group(1)), "") if match else (1, 0, code)

    def _column_group(self, column: str) -> Optional[str]:
        if column in self.BREED_COLS:
            return "breed"
        if column in self.GENDER_COLS:
            return "gender"
        if column in self.WEIGHT_COLS:
            return "weight"
        if column in self.AGE_COLS:
            return "age"
        if column in self.symptom_cols:
            return "symptom"
        return None

    def _disease_cell(
        self, disease: Dict[str, Any], column: str, group: str,
    ) -> int:
        if group == "breed":
            return 1 if column in disease.get("breeds", []) else 0
        if group == "gender":
            return 1 if column in disease.get("genders", []) else 0
        if group == "weight":
            return int(disease.get("weights", {}).get(column, 0))
        if group == "age":
            return int(disease.get("ages", {}).get(column, 0))
        return int(disease.get("symptoms", {}).get(column, 0))

    def get_table(self) -> Dict[str, Any]:
        """Return the matrix as a dense table mirroring the CSV layout.

        Unlike :attr:`matrix_data` (sparse, zeros dropped) this keeps every
        cell so the dashboard can render the exact grid found in the CSV.
        """
        columns: List[Dict[str, str]] = []
        for group, cols in (
            ("breed", self.BREED_COLS),
            ("gender", self.GENDER_COLS),
            ("weight", self.WEIGHT_COLS),
            ("age", self.AGE_COLS),
            ("symptom", self.symptom_cols),
        ):
            for col in cols:
                columns.append(
                    {
                        "key": col,
                        "group": group,
                        "kind": (
                            "binary"
                            if group in ("breed", "gender")
                            else "weight"
                        ),
                    }
                )

        rows = [
            {
                "disease": disease_id,
                "values": {
                    col["key"]: self._disease_cell(
                        disease, col["key"], col["group"],
                    )
                    for col in columns
                },
            }
            for disease_id, disease in self.matrix_data.get(
                "diseases", {},
            ).items()
        ]

        return {
            "columns": columns,
            "rows": rows,
            "id_column": "DISEASES",
            "value_range": {"min": self.CELL_MIN, "max": self.CELL_MAX},
            "csv_path": os.path.abspath(self.csv_path),
        }

    def set_cell(
        self,
        disease_id: str,
        column: str,
        value: int,
        save: bool = True,
    ) -> Dict[str, Any]:
        """Write a single cell and persist the CSV.

        Breed/gender cells are binary (0/1); every other cell accepts an
        integer in ``[CELL_MIN, CELL_MAX]``.
        """
        disease_id = (disease_id or "").strip()
        column = (column or "").strip()

        disease = self.matrix_data.get("diseases", {}).get(disease_id)
        if disease is None:
            raise KeyError(
                f"Không tìm thấy bệnh '{disease_id}' trong ma trận."
            )

        group = self._column_group(column)
        if group is None:
            raise KeyError(
                f"Không tìm thấy cột '{column}' trong ma trận."
            )

        value = int(value)

        if group in ("breed", "gender"):
            if value not in (0, 1):
                raise ValueError(
                    f"Cột '{column}' chỉ nhận giá trị 0 hoặc 1."
                )
            bucket = "breeds" if group == "breed" else "genders"
            current = disease.setdefault(bucket, [])
            if value == 1 and column not in current:
                current.append(column)
            elif value == 0 and column in current:
                current.remove(column)
        else:
            if not self.CELL_MIN <= value <= self.CELL_MAX:
                raise ValueError(
                    f"Giá trị phải nằm trong khoảng "
                    f"{self.CELL_MIN}..{self.CELL_MAX}."
                )
            bucket = {
                "weight": "weights",
                "age": "ages",
                "symptom": "symptoms",
            }[group]
            store = disease.setdefault(bucket, {})
            if value == 0:
                store.pop(column, None)
            else:
                store[column] = value

        if save:
            self._save_matrix()

        return {
            "disease": disease_id,
            "column": column,
            "group": group,
            "value": value,
        }

    def set_cells(
        self, updates: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Apply several cell updates, writing the CSV once at the end."""
        results = [
            self.set_cell(
                disease_id=item.get("disease", ""),
                column=item.get("column", ""),
                value=int(item.get("value", 0)),
                save=False,
            )
            for item in updates
        ]
        self._save_matrix()
        return results

    def add_disease_row(self, disease_id: str, save: bool = True) -> bool:
        """Add an all-zero row for *disease_id*. False if it already existed."""
        disease_id = (disease_id or "").strip()
        if not disease_id:
            raise ValueError("Mã bệnh không được để trống.")

        diseases = self.matrix_data.setdefault("diseases", {})
        if disease_id in diseases:
            return False

        diseases[disease_id] = {
            "breeds": [],
            "genders": [],
            "weights": {},
            "ages": {},
            "symptoms": {},
        }

        if save:
            self._save_matrix()
        return True

    def remove_disease_row(self, disease_id: str, save: bool = True) -> bool:
        """Drop a disease row. Returns False when it was not present."""
        diseases = self.matrix_data.get("diseases", {})
        disease_id = (disease_id or "").strip()
        if disease_id not in diseases:
            return False

        diseases.pop(disease_id)
        if save:
            self._save_matrix()
        return True

    def add_symptom_column(self, symptom_id: str, save: bool = True) -> bool:
        """Add a zero-filled symptom column, keeping SYnnn order."""
        symptom_id = (symptom_id or "").strip().upper()
        if not symptom_id:
            raise ValueError("Mã triệu chứng không được để trống.")

        metadata = self.matrix_data.setdefault("metadata", {})
        symptoms = metadata.setdefault("symptoms", [])
        if symptom_id in symptoms:
            return False

        symptoms.append(symptom_id)
        symptoms.sort(key=self._symptom_sort_key)

        if save:
            self._save_matrix()
        return True

    def remove_symptom_column(
        self, symptom_id: str, save: bool = True,
    ) -> bool:
        """Drop a symptom column from the metadata and from every disease."""
        symptom_id = (symptom_id or "").strip().upper()
        metadata = self.matrix_data.setdefault("metadata", {})
        symptoms = metadata.setdefault("symptoms", [])

        if symptom_id not in symptoms:
            return False

        symptoms.remove(symptom_id)
        for disease in self.matrix_data.get("diseases", {}).values():
            disease.get("symptoms", {}).pop(symptom_id, None)

        if save:
            self._save_matrix()
        return True

    def sync_with_catalog(
        self,
        symptom_ids: Optional[List[str]] = None,
        disease_ids: Optional[List[str]] = None,
    ) -> Dict[str, List[str]]:
        """Add matrix columns/rows for catalog entries that are missing.

        Add-only on purpose: removals go through the explicit delete
        endpoints so an out-of-sync database can never silently wipe
        hand-authored weights.
        """
        added_symptoms = [
            code
            for code in (symptom_ids or [])
            if self.add_symptom_column(code, save=False)
        ]
        added_diseases = [
            code
            for code in (disease_ids or [])
            if self.add_disease_row(code, save=False)
        ]

        if added_symptoms or added_diseases:
            self._save_matrix()
            logger.info(
                "Matrix synced with catalog (+%d symptoms, +%d diseases)",
                len(added_symptoms),
                len(added_diseases),
            )

        return {"symptoms": added_symptoms, "diseases": added_diseases}

    def reload(self) -> None:
        """Re-read the CSV from disk, discarding the in-memory copy."""
        self._load_matrix()
