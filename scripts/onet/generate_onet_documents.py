"""Generate full and section-level O*NET JSONL documents for RAG retrieval.

This script reads the imported DuckDB O*NET tables, assembles occupation
profiles with tasks, skills, knowledge, abilities, software, education, and
related sections, then writes JSONL documents under ``data/documents``.
"""

import duckdb
import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

# Database connection
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = PROJECT_ROOT / "data" / "duckdb" / "onet.duckdb"
OUTPUT_DIR = PROJECT_ROOT / "data" / "documents"

# Ensure output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FULL_DOCUMENTS_PATH = OUTPUT_DIR / "onet_occupation_documents.jsonl"
SECTION_DOCUMENTS_PATH = OUTPUT_DIR / "onet_occupation_section_documents.jsonl"


def get_conn():
    """Get DuckDB connection."""
    return duckdb.connect(str(DB_PATH), read_only=True)


def get_top_skills(conn, onet_code: str, limit: int = 20) -> List[Tuple[str, str, float]]:
    """Get top essential skills. Returns list of (element_id, element_name, data_value)."""
    result = conn.execute("""
        SELECT 
            es.element_id,
            cmr.element_name,
            es.data_value
        FROM essential_skills es
        JOIN content_model_reference cmr ON es.element_id = cmr.element_id
        WHERE es.onetsoc_code = ?
            AND es.scale_id = 'IM'
        ORDER BY es.data_value DESC
        LIMIT ?
    """, [onet_code, limit]).fetchall()
    return result


def get_transferable_skills(conn, onet_code: str, limit: int = 20) -> List[Tuple[str, str, float]]:
    """Get top transferable skills."""
    result = conn.execute("""
        SELECT 
            ts.element_id,
            cmr.element_name,
            ts.data_value
        FROM transferable_skills ts
        JOIN content_model_reference cmr ON ts.element_id = cmr.element_id
        WHERE ts.onetsoc_code = ?
            AND ts.scale_id = 'IM'
        ORDER BY ts.data_value DESC
        LIMIT ?
    """, [onet_code, limit]).fetchall()
    return result


def get_knowledge(conn, onet_code: str, limit: int = 20) -> List[Tuple[str, str, float]]:
    """Get top knowledge areas."""
    result = conn.execute("""
        SELECT 
            k.element_id,
            cmr.element_name,
            k.data_value
        FROM knowledge k
        JOIN content_model_reference cmr ON k.element_id = cmr.element_id
        WHERE k.onetsoc_code = ?
            AND k.scale_id = 'IM'
        ORDER BY k.data_value DESC
        LIMIT ?
    """, [onet_code, limit]).fetchall()
    return result


def get_abilities(conn, onet_code: str, limit: int = 20) -> List[Tuple[str, str, float]]:
    """Get top abilities."""
    result = conn.execute("""
        SELECT 
            a.element_id,
            cmr.element_name,
            a.data_value
        FROM abilities a
        JOIN content_model_reference cmr ON a.element_id = cmr.element_id
        WHERE a.onetsoc_code = ?
            AND a.scale_id = 'IM'
        ORDER BY a.data_value DESC
        LIMIT ?
    """, [onet_code, limit]).fetchall()
    return result


def get_work_activities(conn, onet_code: str, limit: int = 20) -> List[Tuple[str, str, float]]:
    """Get top work activities."""
    result = conn.execute("""
        SELECT 
            wa.element_id,
            cmr.element_name,
            wa.data_value
        FROM work_activities wa
        JOIN content_model_reference cmr ON wa.element_id = cmr.element_id
        WHERE wa.onetsoc_code = ?
            AND wa.scale_id = 'IM'
        ORDER BY wa.data_value DESC
        LIMIT ?
    """, [onet_code, limit]).fetchall()
    return result


def get_work_styles(conn, onet_code: str, limit: int = 20) -> List[Tuple[str, str, float, str]]:
    """
    Get work styles with deduplication by element_name.
    Prefer scale_id='IM', then 'WI', then highest data_value.
    """
    result = conn.execute("""
        SELECT 
            ws.element_id,
            cmr.element_name,
            ws.data_value,
            ws.scale_id
        FROM work_styles ws
        JOIN content_model_reference cmr ON ws.element_id = cmr.element_id
        WHERE ws.onetsoc_code = ?
        ORDER BY 
            cmr.element_name,
            CASE WHEN ws.scale_id = 'IM' THEN 0 ELSE 1 END,
            CASE WHEN ws.scale_id = 'WI' THEN 0 ELSE 1 END,
            ws.data_value DESC
    """, [onet_code]).fetchall()
    
    # Deduplicate by element_name, keeping first (preferred scale_id)
    seen = set()
    deduped = []
    for element_id, element_name, data_value, scale_id in result:
        if element_name not in seen:
            seen.add(element_name)
            deduped.append((element_id, element_name, data_value, scale_id))
    
    return deduped[:limit]


def get_work_context(conn, onet_code: str, limit: int = 20) -> List[Dict[str, Any]]:
    """Get work context items with category descriptions."""
    result = conn.execute("""
        SELECT 
            wc.element_id,
            cmr.element_name,
            wc.data_value,
            wc.scale_id,
            wc.category,
            wcc.category_description
        FROM work_context wc
        JOIN content_model_reference cmr ON wc.element_id = cmr.element_id
        LEFT JOIN work_context_categories wcc
            ON wc.element_id = wcc.element_id
            AND wc.scale_id = wcc.scale_id
            AND wc.category = wcc.category
        WHERE wc.onetsoc_code = ?
        ORDER BY wc.data_value DESC
        LIMIT ?
    """, [onet_code, limit]).fetchall()
    return result


def get_tasks(conn, onet_code: str, limit: int = 20) -> List[Tuple[int, str, str]]:
    """Get top tasks. Returns (task_id, task_description, task_type)."""
    result = conn.execute("""
        SELECT 
            ts.task_id,
            ts.task,
            ts.task_type
        FROM task_statements ts
        WHERE ts.onetsoc_code = ?
        ORDER BY ts.task_id
        LIMIT ?
    """, [onet_code, limit]).fetchall()
    return result


def get_career_interests(conn, onet_code: str) -> List[Tuple[str, str, float]]:
    """Get career interests. Filter to only RIASEC types."""
    riasec_types = ['Realistic', 'Investigative', 'Artistic', 'Social', 'Enterprising', 'Conventional']
    
    result = conn.execute("""
        SELECT 
            cit.element_id,
            cmr.element_name,
            cit.data_value
        FROM career_interest_types cit
        JOIN content_model_reference cmr ON cit.element_id = cmr.element_id
        WHERE cit.onetsoc_code = ?
        ORDER BY cit.data_value DESC
    """, [onet_code]).fetchall()
    
    # Filter to only RIASEC types
    filtered = [
        (eid, ename, dval) for eid, ename, dval in result
        if any(riasec in ename for riasec in riasec_types)
    ]
    
    return filtered


def get_specific_interests(conn, onet_code: str, limit: int = 20) -> List[Tuple[str, str, float]]:
    """Get specific interest areas."""
    result = conn.execute("""
        SELECT 
            sia.element_id,
            cmr.element_name,
            sia.data_value
        FROM specific_interest_areas sia
        JOIN content_model_reference cmr ON sia.element_id = cmr.element_id
        WHERE sia.onetsoc_code = ?
            AND sia.scale_id = 'IM'
        ORDER BY sia.data_value DESC
        LIMIT ?
    """, [onet_code, limit]).fetchall()
    return result


def get_software_skills(conn, onet_code: str, limit: int = 20) -> List[Dict[str, Any]]:
    """Get software skills. Return workplace_example as primary display."""
    result = conn.execute("""
        SELECT DISTINCT
            workplace_example,
            hot_technology,
            in_demand
        FROM software_skills
        WHERE onetsoc_code = ?
        ORDER BY workplace_example
        LIMIT ?
    """, [onet_code, limit]).fetchall()
    
    # Convert to list of dicts, filtering empty examples
    return [
        {
            "skill": row[0],
            "hot_technology": row[1],
            "in_demand": row[2]
        }
        for row in result
        if row[0] and row[0].strip()  # Only include non-empty skills
    ]


def get_job_zone(conn, onet_code: str) -> Optional[Dict[str, Any]]:
    """Get job zone information."""
    result = conn.execute("""
        SELECT 
            jz.job_zone,
            jzr.name,
            jzr.experience,
            jzr.education,
            jzr.job_training
        FROM job_zones jz
        LEFT JOIN job_zone_reference jzr ON jz.job_zone = jzr.job_zone
        WHERE jz.onetsoc_code = ?
        LIMIT 1
    """, [onet_code]).fetchone()
    
    if result:
        return {
            "zone": result[0],
            "name": result[1],
            "experience": result[2],
            "education": result[3],
            "training": result[4]
        }
    return None


def get_education(conn, onet_code: str) -> List[Dict[str, Any]]:
    """Get education requirements. Join with categories and deduplicate."""
    result = conn.execute("""
        SELECT 
            e.element_id,
            cmr.element_name,
            ec.category_description,
            e.data_value
        FROM education e
        JOIN content_model_reference cmr ON e.element_id = cmr.element_id
        LEFT JOIN education_categories ec 
            ON e.element_id = ec.element_id
            AND e.scale_id = ec.scale_id
            AND e.category = ec.category
        WHERE e.onetsoc_code = ?
        ORDER BY e.data_value DESC
    """, [onet_code]).fetchall()
    
    # Deduplicate by category_description, keep highest data_value
    seen = {}
    for element_id, element_name, category_desc, data_value in result:
        if category_desc and category_desc.strip():
            key = category_desc.strip()
            if key not in seen or data_value > seen[key]['value']:
                seen[key] = {
                    "name": category_desc,
                    "value": data_value
                }
    
    return list(seen.values())


def get_training(conn, onet_code: str) -> List[Dict[str, Any]]:
    """Get training and experience requirements. Join with categories and deduplicate."""
    result = conn.execute("""
        SELECT 
            te.element_id,
            cmr.element_name,
            tec.category_description,
            te.data_value
        FROM training_and_experience te
        JOIN content_model_reference cmr ON te.element_id = cmr.element_id
        LEFT JOIN training_and_experience_categories tec
            ON te.element_id = tec.element_id
            AND te.scale_id = tec.scale_id
            AND te.category = tec.category
        WHERE te.onetsoc_code = ?
        ORDER BY te.data_value DESC
    """, [onet_code]).fetchall()
    
    # Deduplicate by category_description, keep highest data_value
    seen = {}
    for element_id, element_name, category_desc, data_value in result:
        if category_desc and category_desc.strip():
            key = category_desc.strip()
            if key not in seen or data_value > seen[key]['value']:
                seen[key] = {
                    "name": category_desc,
                    "value": data_value
                }
    
    return list(seen.values())


def format_skill_list(skills: List[Tuple[str, str, float]]) -> str:
    """Format skills list as readable text."""
    if not skills:
        return "No data available."
    
    lines = []
    for element_id, element_name, data_value in skills:
        lines.append(f"- {element_name} ({data_value})")
    return "\n".join(lines)


def clean_text_lines(lines: List[str]) -> List[str]:
    """Remove consecutive duplicates and empty lines, preserve section separators."""
    cleaned = []
    seen = set()
    last_was_empty = False
    
    for line in lines:
        if isinstance(line, str):
            line_stripped = line.strip()
            # Keep empty lines only for section separation (max one consecutive empty line)
            if not line_stripped:
                if not last_was_empty:
                    cleaned.append("")
                    last_was_empty = True
            else:
                # Remove duplicate content lines, keep first occurrence
                if line_stripped not in seen:
                    seen.add(line_stripped)
                    cleaned.append(line_stripped)
                    last_was_empty = False
    
    return cleaned


def build_full_document(conn, onet_code: str, title: str, description: str) -> Tuple[Dict[str, Any], List[str]]:
    """
    Build a complete occupation document.
    Returns (document_dict, warnings_list)
    """
    warnings = []
    
    # Get all data
    tasks = get_tasks(conn, onet_code, limit=20)
    essential_skills = get_top_skills(conn, onet_code, limit=10)
    transferable_skills = get_transferable_skills(conn, onet_code, limit=10)
    knowledge = get_knowledge(conn, onet_code, limit=10)
    abilities = get_abilities(conn, onet_code, limit=10)
    work_activities = get_work_activities(conn, onet_code, limit=10)
    work_styles = get_work_styles(conn, onet_code, limit=10)
    work_context = get_work_context(conn, onet_code, limit=10)
    career_interests = get_career_interests(conn, onet_code)
    specific_interests = get_specific_interests(conn, onet_code, limit=10)
    software_skills = get_software_skills(conn, onet_code, limit=15)
    job_zone = get_job_zone(conn, onet_code)
    education = get_education(conn, onet_code)
    training = get_training(conn, onet_code)
    
    # Warnings
    if not tasks:
        warnings.append(f"{onet_code}: No tasks found")
    if not essential_skills:
        warnings.append(f"{onet_code}: No essential skills found")
    if not knowledge:
        warnings.append(f"{onet_code}: No knowledge areas found")
    
    # Build text document
    text_parts = [
        f"Occupation: {title}",
        f"O*NET-SOC Code: {onet_code}",
        "",
        "Description:",
        description,
        ""
    ]
    
    # Tasks
    if tasks:
        text_parts.append("Typical Tasks:")
        for task_id, task_desc, task_type in tasks[:10]:
            text_parts.append(f"- {task_desc}")
        text_parts.append("")
    
    # Essential Skills
    if essential_skills:
        text_parts.append("Essential Skills:")
        for element_id, element_name, data_value in essential_skills:
            text_parts.append(f"- {element_name} ({data_value})")
        text_parts.append("")
    
    # Transferable Skills
    if transferable_skills:
        text_parts.append("Transferable Skills:")
        for element_id, element_name, data_value in transferable_skills:
            text_parts.append(f"- {element_name} ({data_value})")
        text_parts.append("")
    
    # Knowledge
    if knowledge:
        text_parts.append("Knowledge Areas:")
        for element_id, element_name, data_value in knowledge:
            text_parts.append(f"- {element_name} ({data_value})")
        text_parts.append("")
    
    # Abilities
    if abilities:
        text_parts.append("Abilities:")
        for element_id, element_name, data_value in abilities:
            text_parts.append(f"- {element_name} ({data_value})")
        text_parts.append("")
    
    # Work Activities
    if work_activities:
        text_parts.append("Work Activities:")
        for element_id, element_name, data_value in work_activities:
            text_parts.append(f"- {element_name} ({data_value})")
        text_parts.append("")
    
    # Work Styles
    if work_styles:
        text_parts.append("Work Styles:")
        for element_id, element_name, data_value, scale_id in work_styles:
            text_parts.append(f"- {element_name} ({data_value}, {scale_id})")
        text_parts.append("")
    
    # Work Context
    if work_context:
        text_parts.append("Work Context:")
        for row in work_context[:10]:
            element_id, element_name, data_value, scale_id, category, category_desc = row
            if category_desc:
                text_parts.append(f"- {element_name}: {category_desc}")
            else:
                text_parts.append(f"- {element_name}")
        text_parts.append("")
    
    # Career Interests
    if career_interests:
        text_parts.append("Career Interests:")
        for element_id, element_name, data_value in career_interests:
            text_parts.append(f"- {element_name} ({data_value})")
        text_parts.append("")
    
    # Specific Interest Areas
    if specific_interests:
        text_parts.append("Specific Interest Areas:")
        for element_id, element_name, data_value in specific_interests[:10]:
            text_parts.append(f"- {element_name} ({data_value})")
        text_parts.append("")
    
    # Software Skills
    if software_skills:
        text_parts.append("Software Skills:")
        for skill_info in software_skills[:15]:
            skill = skill_info["skill"]
            in_demand = skill_info["in_demand"]
            demand_label = "(In Demand)" if in_demand == "Y" else ""
            if demand_label:
                text_parts.append(f"- {skill} {demand_label}")
            else:
                text_parts.append(f"- {skill}")
        text_parts.append("")
    
    # Job Zone
    if job_zone:
        text_parts.append(f"Job Zone: {job_zone['name']} (Zone {job_zone['zone']})")
        if job_zone['experience']:
            text_parts.append(f"Experience: {job_zone['experience']}")
        if job_zone['education']:
            text_parts.append(f"Education: {job_zone['education']}")
        if job_zone['training']:
            text_parts.append(f"Training: {job_zone['training']}")
        text_parts.append("")
    
    # Education
    if education:
        text_parts.append("Education:")
        for edu in education[:5]:
            text_parts.append(f"- {edu['name']} ({edu['value']:.2f})")
        text_parts.append("")
    
    # Training
    if training:
        text_parts.append("Training and Experience:")
        for train in training[:5]:
            text_parts.append(f"- {train['name']} ({train['value']:.2f})")
        text_parts.append("")
    
    text_parts.append("Source: O*NET 30.3 Database")
    
    # Clean and join text
    text_parts = clean_text_lines(text_parts)
    full_text = "\n".join(text_parts)
    
    doc = {
        "id": f"onet-{onet_code}-full",
        "text": full_text,
        "metadata": {
            "source": "O*NET 30.3",
            "source_type": "onet_full_occupation",
            "onet_soc_code": onet_code,
            "occupation_title": title
        }
    }
    
    return doc, warnings


def build_section_documents(conn, onet_code: str, title: str) -> List[Dict[str, Any]]:
    """Build section-level documents for an occupation."""
    docs = []
    
    # Skills section
    essential_skills = get_top_skills(conn, onet_code, limit=20)
    transferable_skills = get_transferable_skills(conn, onet_code, limit=20)
    
    if essential_skills or transferable_skills:
        text_parts = [f"{title} - Skills", f"O*NET-SOC Code: {onet_code}", ""]
        text_parts.append("Essential Skills (Importance):")
        for element_id, element_name, data_value in essential_skills:
            text_parts.append(f"- {element_name}: {data_value}")
        text_parts.append("")
        text_parts.append("Transferable Skills (Importance):")
        for element_id, element_name, data_value in transferable_skills:
            text_parts.append(f"- {element_name}: {data_value}")
        
        text_parts = clean_text_lines(text_parts)
        docs.append({
            "id": f"onet-{onet_code}-skills",
            "text": "\n".join(text_parts),
            "metadata": {
                "source": "O*NET 30.3",
                "source_type": "onet_section",
                "section": "Skills",
                "onet_soc_code": onet_code,
                "occupation_title": title
            }
        })
    
    # Knowledge section
    knowledge = get_knowledge(conn, onet_code, limit=20)
    if knowledge:
        text_parts = [f"{title} - Knowledge", f"O*NET-SOC Code: {onet_code}", ""]
        for element_id, element_name, data_value in knowledge:
            text_parts.append(f"- {element_name}: {data_value}")
        
        text_parts = clean_text_lines(text_parts)
        docs.append({
            "id": f"onet-{onet_code}-knowledge",
            "text": "\n".join(text_parts),
            "metadata": {
                "source": "O*NET 30.3",
                "source_type": "onet_section",
                "section": "Knowledge",
                "onet_soc_code": onet_code,
                "occupation_title": title
            }
        })
    
    # Abilities section
    abilities = get_abilities(conn, onet_code, limit=20)
    if abilities:
        text_parts = [f"{title} - Abilities", f"O*NET-SOC Code: {onet_code}", ""]
        for element_id, element_name, data_value in abilities:
            text_parts.append(f"- {element_name}: {data_value}")
        
        text_parts = clean_text_lines(text_parts)
        docs.append({
            "id": f"onet-{onet_code}-abilities",
            "text": "\n".join(text_parts),
            "metadata": {
                "source": "O*NET 30.3",
                "source_type": "onet_section",
                "section": "Abilities",
                "onet_soc_code": onet_code,
                "occupation_title": title
            }
        })
    
    # Tasks section
    tasks = get_tasks(conn, onet_code, limit=30)
    if tasks:
        text_parts = [f"{title} - Tasks", f"O*NET-SOC Code: {onet_code}", ""]
        for task_id, task_desc, task_type in tasks:
            text_parts.append(f"- {task_desc}")
        
        text_parts = clean_text_lines(text_parts)
        docs.append({
            "id": f"onet-{onet_code}-tasks",
            "text": "\n".join(text_parts),
            "metadata": {
                "source": "O*NET 30.3",
                "source_type": "onet_section",
                "section": "Tasks",
                "onet_soc_code": onet_code,
                "occupation_title": title
            }
        })
    
    # Work Activities section
    work_activities = get_work_activities(conn, onet_code, limit=20)
    if work_activities:
        text_parts = [f"{title} - Work Activities", f"O*NET-SOC Code: {onet_code}", ""]
        for element_id, element_name, data_value in work_activities:
            text_parts.append(f"- {element_name}: {data_value}")
        
        text_parts = clean_text_lines(text_parts)
        docs.append({
            "id": f"onet-{onet_code}-work-activities",
            "text": "\n".join(text_parts),
            "metadata": {
                "source": "O*NET 30.3",
                "source_type": "onet_section",
                "section": "Work Activities",
                "onet_soc_code": onet_code,
                "occupation_title": title
            }
        })
    
    # Work Context section
    work_context = get_work_context(conn, onet_code, limit=20)
    if work_context:
        text_parts = [f"{title} - Work Context", f"O*NET-SOC Code: {onet_code}", ""]
        for row in work_context:
            element_id, element_name, data_value, scale_id, category, category_desc = row
            if category_desc:
                text_parts.append(f"- {element_name}: {category_desc}")
            else:
                text_parts.append(f"- {element_name}")
        
        text_parts = clean_text_lines(text_parts)
        docs.append({
            "id": f"onet-{onet_code}-work-context",
            "text": "\n".join(text_parts),
            "metadata": {
                "source": "O*NET 30.3",
                "source_type": "onet_section",
                "section": "Work Context",
                "onet_soc_code": onet_code,
                "occupation_title": title
            }
        })
    
    # Work Styles section
    work_styles = get_work_styles(conn, onet_code, limit=20)
    if work_styles:
        text_parts = [f"{title} - Work Styles", f"O*NET-SOC Code: {onet_code}", ""]
        for element_id, element_name, data_value, scale_id in work_styles:
            text_parts.append(f"- {element_name}: {data_value} ({scale_id})")
        
        text_parts = clean_text_lines(text_parts)
        docs.append({
            "id": f"onet-{onet_code}-work-styles",
            "text": "\n".join(text_parts),
            "metadata": {
                "source": "O*NET 30.3",
                "source_type": "onet_section",
                "section": "Work Styles",
                "onet_soc_code": onet_code,
                "occupation_title": title
            }
        })
    
    # Interests section
    career_interests = get_career_interests(conn, onet_code)
    specific_interests = get_specific_interests(conn, onet_code, limit=20)
    
    if career_interests or specific_interests:
        text_parts = [f"{title} - Interests", f"O*NET-SOC Code: {onet_code}", ""]
        text_parts.append("Career Interests:")
        for element_id, element_name, data_value in career_interests:
            text_parts.append(f"- {element_name}: {data_value}")
        text_parts.append("")
        text_parts.append("Specific Interest Areas:")
        for element_id, element_name, data_value in specific_interests:
            text_parts.append(f"- {element_name}: {data_value}")
        
        text_parts = clean_text_lines(text_parts)
        docs.append({
            "id": f"onet-{onet_code}-interests",
            "text": "\n".join(text_parts),
            "metadata": {
                "source": "O*NET 30.3",
                "source_type": "onet_section",
                "section": "Interests",
                "onet_soc_code": onet_code,
                "occupation_title": title
            }
        })
    
    # Software Skills section
    software_skills = get_software_skills(conn, onet_code, limit=30)
    if software_skills:
        text_parts = [f"{title} - Software Skills", f"O*NET-SOC Code: {onet_code}", ""]
        for skill_info in software_skills:
            skill = skill_info["skill"]
            in_demand = skill_info["in_demand"]
            if in_demand == "Y":
                text_parts.append(f"- {skill} (In Demand)")
            else:
                text_parts.append(f"- {skill}")
        
        docs.append({
            "id": f"onet-{onet_code}-software",
            "text": "\n".join(text_parts),
            "metadata": {
                "source": "O*NET 30.3",
                "source_type": "onet_section",
                "section": "Software Skills",
                "onet_soc_code": onet_code,
                "occupation_title": title
            }
        })
    
    return docs


def write_jsonl(filepath: Path, documents: List[Dict[str, Any]]) -> None:
    """Write documents to JSONL file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        for doc in documents:
            json.dump(doc, f, ensure_ascii=False)
            f.write('\n')


def main():
    """Main function."""
    conn = get_conn()
    
    print("=" * 80)
    print("GENERATING O*NET RAG DOCUMENTS")
    print("=" * 80)
    
    # Get all occupations
    occupations = conn.execute("""
        SELECT onetsoc_code, title, description FROM occupation_data
        ORDER BY onetsoc_code
    """).fetchall()
    
    print(f"Found {len(occupations)} occupations\n")
    
    full_documents = []
    section_documents = []
    all_warnings = []
    example_doc = None
    
    # Process each occupation
    for idx, (onet_code, title, description) in enumerate(occupations):
        if (idx + 1) % 100 == 0:
            print(f"  Processing occupation {idx + 1}/{len(occupations)}: {onet_code}")
        
        # Build full document
        full_doc, warnings = build_full_document(conn, onet_code, title, description or "")
        full_documents.append(full_doc)
        all_warnings.extend(warnings)
        
        # Save first document as example
        if example_doc is None:
            example_doc = full_doc
        
        # Build section documents
        section_docs = build_section_documents(conn, onet_code, title)
        section_documents.extend(section_docs)
    
    print(f"\nWriting full occupation documents to {FULL_DOCUMENTS_PATH}...")
    write_jsonl(FULL_DOCUMENTS_PATH, full_documents)
    
    print(f"Writing section documents to {SECTION_DOCUMENTS_PATH}...")
    write_jsonl(SECTION_DOCUMENTS_PATH, section_documents)
    
    # Report
    print("\n" + "=" * 80)
    print("GENERATION COMPLETE")
    print("=" * 80)
    print(f"\nStatistics:")
    print(f"  Total occupations processed: {len(occupations)}")
    print(f"  Full documents generated: {len(full_documents)}")
    print(f"  Section documents generated: {len(section_documents)}")
    
    if all_warnings:
        print(f"\nWarnings ({len(all_warnings)}):")
        for warning in all_warnings[:10]:  # Show first 10
            print(f"  - {warning}")
        if len(all_warnings) > 10:
            print(f"  ... and {len(all_warnings) - 10} more")
    
    # Print example
    print("\n" + "=" * 80)
    print("EXAMPLE FULL DOCUMENT")
    print("=" * 80)
    if example_doc:
        print(f"\nID: {example_doc['id']}")
        print(f"Metadata: {example_doc['metadata']}")
        print(f"\nText (first 1000 chars):")
        print(example_doc['text'][:1000])
        print("\n...")
    
    conn.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
