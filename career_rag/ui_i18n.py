"""Small UI translation helpers for the Streamlit career app."""

from __future__ import annotations

import os
import re
from functools import lru_cache
from typing import Any

from career_rag.config import ENV_PATH


DEFAULT_LANGUAGE = "en"
LANGUAGE_SESSION_KEY = "ui_language"
SUPPORTED_LANGUAGES = ("en", "tr")
LANGUAGE_BUTTON_LABELS = {
    "en": "ENG",
    "tr": "TUR",
}


TEXT: dict[str, dict[str, str]] = {
    "en": {
        "language.label": "Language",
        "app.title": "AI Aware Career Guide",
        "app.caption": (
            "A personalized career guidance tool that uses O*NET job data, open-ended "
            "follow-up questions [1] [2], and AI exposure data mainly from Anthropic [3] "
            "and NBER [4] to recommend careers with context about automation and future impact."
        ),
        "app.method_references": "Method references",
        "initial.instructions": (
            "Check the box by the activities you would like to do. Do not think about "
            "how much education/training is needed or how much money you will make."
        ),
        "initial.activities_header": "60 Work Activities",
        "initial.job_zone_header": "Job Zone Questions",
        "initial.current_job_zone_heading": "A. Current Job Zone",
        "initial.current_job_zone_question": (
            "What level of education, training, and experience do you currently have?"
        ),
        "initial.future_job_zone_heading": "B. Future Job Zone",
        "initial.future_job_zone_question": (
            "What level of education, training, and experience are you willing to work toward?"
        ),
        "initial.submit": "Submit",
        "followup.header": "Follow-Up Refinement",
        "followup.ambiguity_info": (
            "Your profile has some close or tied interest areas. You can answer a few "
            "follow-up questions to refine your result."
        ),
        "followup.clear_info": (
            "Your scores are clear, but you can answer a few questions to personalize "
            "the recommendations further."
        ),
        "followup.continue": "Continue with follow-up questions",
        "followup.skip": "Skip follow-up and use initial result",
        "followup.question_progress": "Question {current} of max {maximum}",
        "followup.answer_label": "Your answer",
        "followup.submit_answer": "Submit answer",
        "followup.empty_answer_warning": "Please add a short answer before continuing.",
        "followup.invalid_json_warning": (
            "The LLM response was not valid JSON after repair. A conservative fallback was saved."
        ),
        "followup.rag_error_prefix": "Could not compute follow-up RAG refinement: ",
        "final.header": "Final Career Report",
        "final.description": (
            "Generate a source-grounded report from the saved Interest Profiler profile, "
            "local O*NET data, and available AI-impact evidence."
        ),
        "final.finish_followup_info": (
            "Finish the follow-up questions to compute the refined ranking before generating the final report."
        ),
        "final.generate_button": "Generate Final Career Report",
        "final.spinner": "Retrieving O*NET and AI-impact evidence...",
        "final.validation_failed_prefix": "Final report validation failed: ",
        "final.generation_error_prefix": "Could not generate the final career report: ",
        "final.success": "Final career report generated.",
        "final.stale_warning_prefix": (
            "Hidden stale final report because it does not match the current profile: "
        ),
        "artifacts.missing_title": "RAG artifacts missing. Please run the setup/build command.",
        "artifacts.missing_list": "Missing or unusable artifacts:",
        "artifacts.use_prebuilt": "Use prebuilt artifacts when available:",
        "artifacts.rebuild": "Or rebuild artifacts from raw/local sources:",
        "artifacts.verify": "Then verify:",
        "report.no_final": "No final career report is available yet.",
        "report.tab.summary": "Summary",
        "report.tab.top_matches": "Top Matches",
        "report.tab.alternatives": "Alternatives",
        "report.tab.semantic": "Semantic Report",
        "report.tab.sources": "Sources",
        "report.summary.header": "Profile Summary",
        "report.summary.top_interests": "Top interests",
        "report.summary.final_code": "Final code",
        "report.summary.current_zone": "Current zone",
        "report.summary.future_zone": "Future zone",
        "report.summary.preferences": "Preferences Used For Matching",
        "report.top_matches.header": "Top Career Matches",
        "report.top_matches.caption": (
            "This report shows up to five careers for your current Job Zone and up to five "
            "careers for the future Job Zone you are willing to work toward."
        ),
        "report.top_matches.fit": "Fit",
        "report.top_matches.detail_card": "Job detail card",
        "report.job.what_does": "What This Job Does",
        "report.job.no_description": "No local O*NET description was retrieved.",
        "report.job.main_tasks": "Main Tasks",
        "report.job.key_skills": "Key Skills",
        "report.job.education_zone": "Education / Job Zone",
        "report.job.ai_impact": "AI Impact Breakdown",
        "report.job.day_life": "Day In The Life",
        "report.ai.table.task": "Task",
        "report.ai.table.signal": "AI exposure signal",
        "report.ai.table.score": "Score",
        "report.ai.table.source": "Source",
        "report.ai.caption": (
            "AI exposure signal is a plain-language level from the local task evidence. "
            "Score represents Observed Claude Usage Share as a percentage, not a "
            "0-100 risk score. For example, a score of 0.4015 means 0.4015% of "
            "mapped Claude conversations, approximately 4 out of every 1,000. "
            "N/A means no exact local task match, not zero AI impact."
        ),
        "report.ai.no_rows": "No AI impact rows were available for this occupation.",
        "report.alternatives.header": "Alternative Careers",
        "report.alternatives.none": "No alternatives were resolved from local O*NET evidence.",
        "report.semantic.header": "Semantic Retrieval Report",
        "report.semantic.need_new_report": (
            "Generate a new final report to add the semantic O*NET and AI-impact comparison."
        ),
        "report.semantic.caption": (
            "This comparison uses follow-up answers and preferences as the strongest semantic "
            "search signal, with the selected current and future Job Zones used to keep results "
            "aligned with the user's education/preparation choices. It is separate from the "
            "scoring-based top matches."
        ),
        "report.semantic.not_enough": (
            "Semantic O*NET/AI-impact retrieval did not return enough evidence."
        ),
        "report.semantic.relevant_heading": "Relevant Careers",
        "report.semantic.technology_heading": "Role of Technology and AI",
        "report.semantic.takeaways_heading": "Takeaways",
        "report.semantic.signals_heading": "Careers Surfaced By Semantic Retrieval",
        "report.semantic.signals_caption": (
            "Similarity signal is a weighted average of Chroma similarity scores. "
            "Rows retrieved from follow-up-answer queries count more than broad profile rows; "
            "O*NET evidence has weight 2 and AI-impact evidence has weight 1. Careers outside "
            "the selected Job Zones are penalized."
        ),
        "report.semantic.table.career": "Career",
        "report.semantic.table.soc": "O*NET-SOC",
        "report.semantic.table.job_zone": "Job Zone",
        "report.semantic.table.signal": "Similarity signal",
        "report.semantic.table.sources": "Sources",
        "report.semantic.retrieved_onet": "Retrieved O*NET evidence",
        "report.semantic.retrieved_ai": "Retrieved AI-impact evidence",
        "report.semantic.table.source": "Source",
        "report.semantic.table.career_section": "Career / section",
        "report.semantic.table.passage": "Retrieved passage",
        "report.semantic.table.occupation_signal": "Occupation / signal",
        "report.semantic.table.task": "Task",
        "report.sources.header": "Sources",
        "report.sources.expander": "View numbered citations",
        "report.no_local_evidence": "No local evidence was retrieved for this section.",
        "report.skills.software": "Software / technical tools",
        "report.skills.technical": "Technical and domain skills",
        "report.skills.foundational": "Foundational communication",
        "report.skills.knowledge": "Knowledge areas",
        "report.preferences.tasks": "Tasks and tools",
        "report.preferences.work_style": "Work style",
        "report.preferences.people": "People and setting",
        "report.preferences.direction": "Career direction",
        "report.preferences.other": "Other",
        "report.education.typical": "Typical preparation: ",
        "report.education.training": "Experience and training: ",
        "report.education.responses": "O*NET education responses",
        "report.education.none": "No local education or Job Zone evidence was retrieved for this section.",
        "report.job_zone.unavailable": "Job Zone unavailable",
        "report.job_zone.preparation_unavailable": "preparation level unavailable",
        "report.job_zone.title": "Job Zone {zone} ({explanation})",
        "report.job_zone.typical_education": "typical education: {education}",
        "report.job_zone.training": "training: {training}",
        "report.career_fallback": "Career",
        "report.education.master_with_bachelor": (
            "Most O*NET responses point to a master's degree for this role; "
            "bachelor's-degree paths are present but less common."
        ),
        "report.education.master": "Most O*NET responses point to a master's degree for this role.",
        "report.education.bachelor_with_master": (
            "Most O*NET responses point to a bachelor's degree; master's-level preparation is also common."
        ),
        "report.education.bachelor": "Most O*NET responses point to a bachelor's degree for this role.",
        "report.education.doctoral": (
            "Most O*NET responses point to doctoral or professional-degree preparation for this role."
        ),
        "report.education.associate": (
            "Most O*NET responses point to associate-degree or vocational preparation for this role."
        ),
        "report.education.high_school": (
            "Most O*NET responses point to high-school-level preparation for this role."
        ),
        "report.education.dynamic": "Most O*NET responses point to {label} for this role.",
        "common.na": "N/A",
        "translation.unavailable": (
            "Dynamic report text is shown in English because OPENAI_API_KEY is not configured "
            "for display-time translation."
        ),
    },
    "tr": {
        "language.label": "Dil",
        "app.title": "Yapay Zeka Farkındalıklı Kariyer Rehberi",
        "app.caption": (
            "O*NET meslek verilerini, açık uçlu takip sorularını [1] [2] ve ağırlıklı olarak "
            "Anthropic [3] ile NBER [4] kaynaklı yapay zeka etkisi verilerini kullanarak "
            "otomasyon ve gelecekteki etki bağlamıyla kariyer önerileri sunan kişiselleştirilmiş "
            "bir kariyer rehberliği aracı."
        ),
        "app.method_references": "Yöntem kaynakları",
        "initial.instructions": (
            "Yapmak isteyeceğin etkinliklerin yanındaki kutuyu işaretle. Ne kadar eğitim/"
            "hazırlık gerektiğini veya ne kadar para kazanacağını düşünme."
        ),
        "initial.activities_header": "60 İş Etkinliği",
        "initial.job_zone_header": "İş Bölgesi Soruları",
        "initial.current_job_zone_heading": "A. Mevcut İş Bölgesi",
        "initial.current_job_zone_question": (
            "Şu anda hangi eğitim, hazırlık ve deneyim düzeyine sahipsin?"
        ),
        "initial.future_job_zone_heading": "B. Hedeflenen İş Bölgesi",
        "initial.future_job_zone_question": (
            "Hangi eğitim, hazırlık ve deneyim düzeyi için çalışmaya isteklisin?"
        ),
        "initial.submit": "Gönder",
        "followup.header": "Takip Sorularıyla İyileştirme",
        "followup.ambiguity_info": (
            "Profilinde birbirine yakın veya eşit çıkan ilgi alanları var. Sonucunu iyileştirmek "
            "için birkaç takip sorusu yanıtlayabilirsin."
        ),
        "followup.clear_info": (
            "Puanların net görünüyor, ancak önerileri daha kişisel hale getirmek için birkaç "
            "soru yanıtlayabilirsin."
        ),
        "followup.continue": "Takip sorularıyla devam et",
        "followup.skip": "Takip sorularını atla ve ilk sonucu kullan",
        "followup.question_progress": "Soru {current} / en fazla {maximum}",
        "followup.answer_label": "Yanıtın",
        "followup.submit_answer": "Yanıtı gönder",
        "followup.empty_answer_warning": "Devam etmeden önce kısa bir yanıt ekle.",
        "followup.invalid_json_warning": (
            "LLM yanıtı onarımdan sonra geçerli JSON değildi. Temkinli bir yedek sonuç kaydedildi."
        ),
        "followup.rag_error_prefix": "Takip sorusu RAG iyileştirmesi hesaplanamadı: ",
        "final.header": "Nihai Kariyer Raporu",
        "final.description": (
            "Kaydedilmiş Interest Profiler profili, yerel O*NET verileri ve mevcut yapay zeka "
            "etkisi kanıtlarından kaynaklara dayalı bir rapor üret."
        ),
        "final.finish_followup_info": (
            "Nihai raporu üretmeden önce iyileştirilmiş sıralamayı hesaplamak için takip "
            "sorularını tamamla."
        ),
        "final.generate_button": "Nihai Kariyer Raporu Oluştur",
        "final.spinner": "O*NET ve yapay zeka etkisi kanıtları getiriliyor...",
        "final.validation_failed_prefix": "Nihai rapor doğrulaması başarısız: ",
        "final.generation_error_prefix": "Nihai kariyer raporu oluşturulamadı: ",
        "final.success": "Nihai kariyer raporu oluşturuldu.",
        "final.stale_warning_prefix": (
            "Geçerli profille eşleşmediği için eski nihai rapor gizlendi: "
        ),
        "artifacts.missing_title": "RAG dosyaları eksik. Lütfen kurulum/oluşturma komutunu çalıştır.",
        "artifacts.missing_list": "Eksik veya kullanılamayan dosyalar:",
        "artifacts.use_prebuilt": "Hazır dosyalar varsa şunu kullan:",
        "artifacts.rebuild": "Ya da ham/yerel kaynaklardan yeniden oluştur:",
        "artifacts.verify": "Ardından doğrula:",
        "report.no_final": "Henüz nihai kariyer raporu yok.",
        "report.tab.summary": "Özet",
        "report.tab.top_matches": "En Uygunlar",
        "report.tab.alternatives": "Alternatifler",
        "report.tab.semantic": "Semantik Rapor",
        "report.tab.sources": "Kaynaklar",
        "report.summary.header": "Profil Özeti",
        "report.summary.top_interests": "En güçlü ilgi alanları",
        "report.summary.final_code": "Nihai kod",
        "report.summary.current_zone": "Mevcut bölge",
        "report.summary.future_zone": "Hedef bölge",
        "report.summary.preferences": "Eşleştirmede Kullanılan Tercihler",
        "report.top_matches.header": "En Uygun Kariyer Eşleşmeleri",
        "report.top_matches.caption": (
            "Bu rapor, mevcut İş Bölgen için en fazla beş kariyeri ve ulaşmaya istekli olduğun "
            "gelecek İş Bölgesi için en fazla beş kariyeri gösterir."
        ),
        "report.top_matches.fit": "Uyum",
        "report.top_matches.detail_card": "İş ayrıntı kartı",
        "report.job.what_does": "Bu İş Ne Yapar",
        "report.job.no_description": "Yerel O*NET açıklaması getirilemedi.",
        "report.job.main_tasks": "Ana Görevler",
        "report.job.key_skills": "Temel Beceriler",
        "report.job.education_zone": "Eğitim / İş Bölgesi",
        "report.job.ai_impact": "Yapay Zeka Etkisi Dağılımı",
        "report.job.day_life": "Günlük İş Akışı",
        "report.ai.table.task": "Görev",
        "report.ai.table.signal": "Yapay zeka maruziyet sinyali",
        "report.ai.table.score": "Puan",
        "report.ai.table.source": "Kaynak",
        "report.ai.caption": (
            "Yapay zeka maruziyet sinyali, yerel görev kanıtlarından türetilmiş sade bir düzeydir. "
            "Puan, 0-100 risk puanı değil, Gözlemlenen Claude Kullanım Payı yüzdesidir. Örneğin "
            "0.4015 puanı, eşlenen Claude konuşmalarının %0.4015'i, yaklaşık her 1.000 konuşmadan "
            "4'ü anlamına gelir. Yok ifadesi, tam yerel görev eşleşmesi bulunmadığı anlamına gelir; "
            "yapay zeka etkisinin sıfır olduğu anlamına gelmez."
        ),
        "report.ai.no_rows": "Bu meslek için yapay zeka etkisi satırı bulunamadı.",
        "report.alternatives.header": "Alternatif Kariyerler",
        "report.alternatives.none": "Yerel O*NET kanıtlarından alternatif çözümlenemedi.",
        "report.semantic.header": "Semantik Getirme Raporu",
        "report.semantic.need_new_report": (
            "Semantik O*NET ve yapay zeka etkisi karşılaştırmasını eklemek için yeni bir nihai rapor oluştur."
        ),
        "report.semantic.caption": (
            "Bu karşılaştırma, takip yanıtlarını ve tercihleri en güçlü semantik arama sinyali "
            "olarak kullanır; seçilen mevcut ve gelecek İş Bölgeleri de sonuçları eğitim/hazırlık "
            "seçimleriyle uyumlu tutmak için kullanılır. Puanlamaya dayalı en iyi eşleşmelerden ayrıdır."
        ),
        "report.semantic.not_enough": (
            "Semantik O*NET/yapay zeka etkisi getirmesi yeterli kanıt döndürmedi."
        ),
        "report.semantic.relevant_heading": "İlgili Kariyerler",
        "report.semantic.technology_heading": "Teknoloji ve Yapay Zekanın Rolü",
        "report.semantic.takeaways_heading": "Çıkarımlar",
        "report.semantic.signals_heading": "Semantik Getirmede Öne Çıkan Kariyerler",
        "report.semantic.signals_caption": (
            "Benzerlik sinyali, Chroma benzerlik puanlarının ağırlıklı ortalamasıdır. Takip "
            "yanıtı sorgularından gelen satırlar geniş profil satırlarından daha fazla ağırlık "
            "taşır; O*NET kanıtı 2, yapay zeka etkisi kanıtı 1 ağırlığındadır. Seçilen İş Bölgeleri "
            "dışındaki kariyerler cezalandırılır."
        ),
        "report.semantic.table.career": "Kariyer",
        "report.semantic.table.soc": "O*NET-SOC",
        "report.semantic.table.job_zone": "İş Bölgesi",
        "report.semantic.table.signal": "Benzerlik sinyali",
        "report.semantic.table.sources": "Kaynaklar",
        "report.semantic.retrieved_onet": "Getirilen O*NET kanıtı",
        "report.semantic.retrieved_ai": "Getirilen yapay zeka etkisi kanıtı",
        "report.semantic.table.source": "Kaynak",
        "report.semantic.table.career_section": "Kariyer / bölüm",
        "report.semantic.table.passage": "Getirilen pasaj",
        "report.semantic.table.occupation_signal": "Meslek / sinyal",
        "report.semantic.table.task": "Görev",
        "report.sources.header": "Kaynaklar",
        "report.sources.expander": "Numaralı atıfları görüntüle",
        "report.no_local_evidence": "Bu bölüm için yerel kanıt getirilemedi.",
        "report.skills.software": "Yazılım / teknik araçlar",
        "report.skills.technical": "Teknik ve alan becerileri",
        "report.skills.foundational": "Temel iletişim",
        "report.skills.knowledge": "Bilgi alanları",
        "report.preferences.tasks": "Görevler ve araçlar",
        "report.preferences.work_style": "Çalışma tarzı",
        "report.preferences.people": "İnsanlar ve ortam",
        "report.preferences.direction": "Kariyer yönü",
        "report.preferences.other": "Diğer",
        "report.education.typical": "Tipik hazırlık: ",
        "report.education.training": "Deneyim ve eğitim: ",
        "report.education.responses": "O*NET eğitim yanıtları",
        "report.education.none": "Bu bölüm için yerel eğitim veya İş Bölgesi kanıtı getirilemedi.",
        "report.job_zone.unavailable": "İş Bölgesi kullanılamıyor",
        "report.job_zone.preparation_unavailable": "hazırlık düzeyi kullanılamıyor",
        "report.job_zone.title": "İş Bölgesi {zone} ({explanation})",
        "report.job_zone.typical_education": "tipik eğitim: {education}",
        "report.job_zone.training": "hazırlık: {training}",
        "report.career_fallback": "Kariyer",
        "report.education.master_with_bachelor": (
            "O*NET yanıtlarının çoğu bu rol için yüksek lisans derecesine işaret ediyor; "
            "lisans derecesi yolları da var ancak daha az yaygın."
        ),
        "report.education.master": "O*NET yanıtlarının çoğu bu rol için yüksek lisans derecesine işaret ediyor.",
        "report.education.bachelor_with_master": (
            "O*NET yanıtlarının çoğu lisans derecesine işaret ediyor; yüksek lisans düzeyi hazırlık da yaygın."
        ),
        "report.education.bachelor": "O*NET yanıtlarının çoğu bu rol için lisans derecesine işaret ediyor.",
        "report.education.doctoral": (
            "O*NET yanıtlarının çoğu bu rol için doktora veya profesyonel derece hazırlığına işaret ediyor."
        ),
        "report.education.associate": (
            "O*NET yanıtlarının çoğu bu rol için ön lisans veya mesleki hazırlığa işaret ediyor."
        ),
        "report.education.high_school": (
            "O*NET yanıtlarının çoğu bu rol için lise düzeyi hazırlığa işaret ediyor."
        ),
        "report.education.dynamic": "O*NET yanıtlarının çoğu bu rol için {label} düzeyine işaret ediyor.",
        "common.na": "Yok",
        "translation.unavailable": (
            "Dinamik rapor metinleri, görüntüleme sırasında çeviri için OPENAI_API_KEY "
            "tanımlanmadığı için İngilizce gösteriliyor."
        ),
    },
}


INTEREST_TRANSLATIONS: dict[str, dict[str, str]] = {
    "en": {
        "Realistic": "Realistic",
        "Investigative": "Investigative",
        "Artistic": "Artistic",
        "Social": "Social",
        "Enterprising": "Enterprising",
        "Conventional": "Conventional",
    },
    "tr": {
        "Realistic": "Gerçekçi",
        "Investigative": "Araştırmacı",
        "Artistic": "Sanatsal",
        "Social": "Sosyal",
        "Enterprising": "Girişimci",
        "Conventional": "Geleneksel",
    },
}


JOB_ZONE_LABEL_TRANSLATIONS: dict[str, dict[int, str]] = {
    "en": {
        1: "Little or No Preparation Needed",
        2: "Some Preparation Needed",
        3: "Medium Preparation Needed",
        4: "Considerable Preparation Needed",
        5: "Extensive Preparation Needed",
    },
    "tr": {
        1: "Az veya Hiç Hazırlık Gerekmez",
        2: "Biraz Hazırlık Gerekir",
        3: "Orta Düzey Hazırlık Gerekir",
        4: "Önemli Ölçüde Hazırlık Gerekir",
        5: "Kapsamlı Hazırlık Gerekir",
    },
}


JOB_ZONE_PREPARATION_TRANSLATIONS: dict[str, dict[int, str]] = {
    "en": {
        1: "no experience required",
        2: "high school diploma required",
        3: "associate's degree or vocational training required",
        4: "bachelor's degree required",
        5: "graduate degree required",
    },
    "tr": {
        1: "deneyim gerekmez",
        2: "lise diploması gerekir",
        3: "ön lisans derecesi veya mesleki eğitim gerekir",
        4: "lisans derecesi gerekir",
        5: "lisansüstü derece gerekir",
    },
}


QUESTION_TRANSLATIONS: dict[str, dict[int, str]] = {
    "tr": {
        1: "Mutfak dolapları yapmak",
        2: "Ofislere ve evlere paket teslim etmek için kamyon kullanmak",
        3: "Tuğla veya karo döşemek",
        4: "Sevkiyattan önce parçaların kalitesini test etmek",
        5: "Ev aletlerini onarmak",
        6: "Kilitleri onarmak ve takmak",
        7: "Balık üretim çiftliğinde balık yetiştirmek",
        8: "Ürün yapmak için makineleri kurmak ve çalıştırmak",
        9: "Elektronik parçaları monte etmek",
        10: "Orman yangınlarını söndürmek",
        11: "Yeni bir ilaç geliştirmek",
        12: "Bir yangının nedenini araştırmak",
        13: "Su kirliliğini azaltmanın yollarını incelemek",
        14: "Hava durumunu daha iyi tahmin etmenin bir yolunu geliştirmek",
        15: "Kimyasal deneyler yapmak",
        16: "Biyoloji laboratuvarında çalışmak",
        17: "Gezegenlerin hareketini incelemek",
        18: "Şekerin yerine geçecek bir madde icat etmek",
        19: "Kan örneklerini mikroskopla incelemek",
        20: "Hastalıkları belirlemek için laboratuvar testleri yapmak",
        21: "Kitaplar veya tiyatro oyunları yazmak",
        22: "Tiyatro oyunları için dekor boyamak",
        23: "Bir müzik aleti çalmak",
        24: "Film veya televizyon programları için senaryo yazmak",
        25: "Müzik bestelemek veya düzenlemek",
        26: "Caz veya tap dansı yapmak",
        27: "Resim çizmek",
        28: "Bir grupta şarkı söylemek",
        29: "Filmler için özel efektler yaratmak",
        30: "Filmleri kurgulamak",
        31: "Bir kişiye egzersiz rutini öğretmek",
        32: "Çocuklara spor yapmayı öğretmek",
        33: "Kişisel veya duygusal sorunları olan insanlara yardım etmek",
        34: "Sağır veya işitme güçlüğü çeken kişilere işaret dili öğretmek",
        35: "İnsanlara kariyer rehberliği yapmak",
        36: "Bir grup terapisi oturumunun yürütülmesine yardım etmek",
        37: "Rehabilitasyon terapisi uygulamak",
        38: "Gündüz bakımevinde çocuklarla ilgilenmek",
        39: "Kar amacı gütmeyen bir kuruluşta gönüllü çalışma yapmak",
        40: "Lise sınıfında ders vermek",
        41: "Hisse senedi ve tahvil alıp satmak",
        42: "İş sözleşmeleri müzakere etmek",
        43: "Bir perakende mağazasını yönetmek",
        44: "Bir davada müvekkili temsil etmek",
        45: "Güzellik salonu veya berber dükkanı işletmek",
        46: "Yeni bir giyim serisini pazarlamak",
        47: "Büyük bir şirkette bir departmanı yönetmek",
        48: "Bir mağazada ürün satmak",
        49: "Kendi işini kurmak",
        50: "Bir giyim mağazasını yönetmek",
        51: "Bilgisayar yazılımı kullanarak elektronik tablo hazırlamak",
        52: "Çalışanların ücretlerini hesaplamak",
        53: "Kayıtları veya formları kontrol edip düzeltmek",
        54: "El bilgisayarı kullanarak malzeme envanteri çıkarmak",
        55: "Büyük bir ağdaki bilgisayarlara yazılım kurmak",
        56: "Kira ödemelerini kaydetmek",
        57: "Hesap makinesi kullanmak",
        58: "Envanter kayıtlarını tutmak",
        59: "Gönderim ve teslim alma kayıtlarını tutmak",
        60: "Bir kuruluş için postaları damgalamak, ayırmak ve dağıtmak",
    }
}


FOLLOWUP_QUESTION_TRANSLATIONS: dict[str, dict[str, str]] = {
    "tr": {
        "work_setting": (
            "Çalışma saatlerini en çok nerede geçirmek isterdin? Fiziksel alanı, atmosferi, "
            "iç mekan mı dış mekan mı olduğunu düşün; ideal çalışma ortamını gözünde canlandırıp anlat."
        ),
        "work_with": (
            "Kendini sevdiğin bir işe tamamen kaptırmış halde hayal ettiğinde aslında ne yapıyorsun? "
            "İş unvanı değil; odada neler oluyor, neyle veya kimlerle etkileşiyorsun?"
        ),
        "independence": (
            "İşte veya okulda kendini en özgür ve etkili hissettiğin bir zamanı anlat. Bunu iyi yapan "
            "neydi: kendi başına mıydın, iş birliği mi yapıyordun, bir yapıyı mı takip ediyordun, "
            "yoksa yolu kendin mi buluyordun?"
        ),
        "team_dynamic": (
            "Günlük olarak başka insanlarla çalışmak sana nasıl geliyor? Enerji mi veriyor, yoruyor mu? "
            "İşinde ideal insan etkileşimi miktarı ve türü nedir?"
        ),
        "impact_type": (
            "Gerçekten iyi bir iş gününün sonunda, yaptığın işin önemli olduğunu sana ne hissettirirdi? "
            "Çalışmanın ne tür bir fark yaratmasını istersin?"
        ),
        "structure_preference": (
            "Gününün net bir ajandası veya planı olmadığında nasıl hissediyorsun? Tersine, her saatin "
            "senin için planlandığında nasıl hissediyorsun?"
        ),
        "skill_anchor": (
            "Okulda, işte, bir projede veya herhangi bir yerde yaptığın ve birinin 'bunda gerçekten iyisin' "
            "dediği, senin de buna inandığın bir şeyi anlat. Ne yapıyordun?"
        ),
        "recognition": (
            "Ne tür bir takdir veya sonuç çalışmanı değerli hissettirir? Görünür bir çıktı mı, uzman "
            "olarak görülmek mi, birinin minnettarlığı mı, yoksa bambaşka bir şey mi?"
        ),
        "future_concern": (
            "Kariyer geleceğini düşündüğünde seni en çok ne endişelendiriyor? Maaş gibi pratik şeyler "
            "değil; yanlış seçim yapmayı hayal ettiğinde ortaya çıkan daha derin korku nedir?"
        ),
        "dream_scenario": (
            "Beş yıl sonra, seni iyi tanıyan biriyle karşılaşıyorsun ve ona neler yaptığını anlatıyorsun. "
            "Ona ne anlatıyor olmak isterdin?"
        ),
    }
}


DISPLAY_TRANSLATIONS: dict[str, dict[str, str]] = {
    "tr": {
        "Current Job Zone options": "Mevcut İş Bölgesi seçenekleri",
        "Future Job Zone options": "Hedeflenen İş Bölgesi seçenekleri",
        "Strong fit": "Güçlü uyum",
        "Good fit": "İyi uyum",
        "Tasks and tools": "Görevler ve araçlar",
        "Work style": "Çalışma tarzı",
        "People and setting": "İnsanlar ve ortam",
        "Career direction": "Kariyer yönü",
        "Other": "Diğer",
        "onet": "O*NET",
        "ai_impact": "Yapay zeka etkisi",
    }
}


ARTIFACT_LABEL_TRANSLATIONS: dict[str, dict[str, str]] = {
    "tr": {
        "onet_duckdb": "O*NET DuckDB veritabanı",
        "onet_full_documents": "O*NET tam meslek dokümanları",
        "onet_section_documents": "O*NET bölüm dokümanları",
        "onet_supplemental_documents": "O*NET ek dokümanları",
        "chroma_onet_dir": "O*NET Chroma deposu",
        "research_documents": "Yapay zeka etkisi araştırma iddiası dokümanları",
        "anthropic_evidence": "Yapılandırılmış Anthropic/yapay zeka etkisi kanıtı",
        "chroma_research_dir": "Araştırma Chroma deposu",
        "chroma_ai_impact_dir": "Yapılandırılmış yapay zeka etkisi Chroma deposu",
    }
}


def normalize_language(language: Any) -> str:
    """Return a supported language code."""
    value = str(language or DEFAULT_LANGUAGE).strip().lower()
    if value in {"tr", "tur", "turkish", "turkce", "türkçe"}:
        return "tr"
    return DEFAULT_LANGUAGE


def ui_text(key: str, language: Any = DEFAULT_LANGUAGE, **format_values: Any) -> str:
    """Return translated UI text, falling back to the exact English string."""
    normalized = normalize_language(language)
    value = TEXT.get(normalized, {}).get(key) or TEXT[DEFAULT_LANGUAGE].get(key) or key
    return value.format(**format_values) if format_values else value


def question_text(question: dict[str, Any], language: Any = DEFAULT_LANGUAGE) -> str:
    """Return the localized Interest Profiler question text for display."""
    normalized = normalize_language(language)
    question_id = question.get("id")
    try:
        numeric_id = int(question_id)
    except (TypeError, ValueError):
        numeric_id = -1
    return QUESTION_TRANSLATIONS.get(normalized, {}).get(numeric_id) or str(question.get("text") or "")


def followup_question_text(question: dict[str, Any], language: Any = DEFAULT_LANGUAGE) -> str:
    """Return the localized follow-up question text for display."""
    normalized = normalize_language(language)
    question_id = str(question.get("id") or "").strip()
    return FOLLOWUP_QUESTION_TRANSLATIONS.get(normalized, {}).get(question_id) or str(
        question.get("question") or ""
    )


def interest_label(interest: Any, language: Any = DEFAULT_LANGUAGE) -> str:
    """Return a localized RIASEC interest name for display."""
    text = str(interest or "")
    normalized = normalize_language(language)
    return INTEREST_TRANSLATIONS.get(normalized, {}).get(text) or text


def job_zone_label(zone: Any, language: Any = DEFAULT_LANGUAGE) -> str:
    """Return a localized O*NET Job Zone label."""
    try:
        zone_int = int(zone)
    except (TypeError, ValueError):
        return str(zone or "")
    normalized = normalize_language(language)
    return JOB_ZONE_LABEL_TRANSLATIONS.get(normalized, {}).get(
        zone_int,
        JOB_ZONE_LABEL_TRANSLATIONS[DEFAULT_LANGUAGE].get(zone_int, str(zone_int)),
    )


def job_zone_preparation(zone: Any, language: Any = DEFAULT_LANGUAGE) -> str:
    """Return a localized preparation explanation for a Job Zone."""
    try:
        zone_int = int(zone)
    except (TypeError, ValueError):
        return ui_text("report.job_zone.preparation_unavailable", language)
    normalized = normalize_language(language)
    return JOB_ZONE_PREPARATION_TRANSLATIONS.get(normalized, {}).get(
        zone_int,
        ui_text("report.job_zone.preparation_unavailable", language),
    )


def display_label(value: Any, language: Any = DEFAULT_LANGUAGE) -> str:
    """Translate known generated labels while leaving unknown data untouched."""
    text = str(value or "")
    normalized = normalize_language(language)
    return DISPLAY_TRANSLATIONS.get(normalized, {}).get(text) or text


def artifact_label(artifact: dict[str, Any], language: Any = DEFAULT_LANGUAGE) -> str:
    """Return a localized artifact label for missing-artifact UI."""
    normalized = normalize_language(language)
    artifact_id = str(artifact.get("id") or "")
    return ARTIFACT_LABEL_TRANSLATIONS.get(normalized, {}).get(artifact_id) or str(
        artifact.get("label") or artifact_id
    )


def dynamic_translation_available(language: Any = DEFAULT_LANGUAGE) -> bool:
    """Return True when dynamic English report text can be translated for the language."""
    if normalize_language(language) == DEFAULT_LANGUAGE:
        return True
    _load_dotenv_if_available()
    return bool(os.getenv("OPENAI_API_KEY"))


def dynamic_text(value: Any, language: Any = DEFAULT_LANGUAGE, context: str = "career report") -> str:
    """Translate dynamic retrieved/generated display text when Turkish UI is active."""
    text = str(value or "").strip()
    if not text:
        return ""
    if normalize_language(language) != "tr":
        return text
    if not _should_translate_dynamic_text(text):
        return text
    return _translate_to_turkish_cached(text, str(context or "career report"))


def dynamic_list(values: Any, language: Any = DEFAULT_LANGUAGE, context: str = "career report") -> list[str]:
    """Translate a sequence of dynamic display strings."""
    if not isinstance(values, (list, tuple, set)):
        return []
    return [dynamic_text(value, language, context) for value in values]


def _should_translate_dynamic_text(text: str) -> bool:
    if not text.strip():
        return False
    if re.fullmatch(r"[\W\d_]+", text):
        return False
    if re.fullmatch(r"\[?\d+\]?|https?://\S+", text):
        return False
    return True


@lru_cache(maxsize=2048)
def _translate_to_turkish_cached(text: str, context: str) -> str:
    _load_dotenv_if_available()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return text
    try:
        from openai import OpenAI
    except ImportError:
        return text

    model = os.getenv("OPENAI_TRANSLATION_MODEL") or os.getenv("OPENAI_MODEL") or "gpt-4o-mini"
    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You translate career-guidance app display text from English to Turkish. "
                        "Preserve citation markers like [1], URLs, O*NET, O*NET-SOC codes, SOC codes, "
                        "percentages, numbers, markdown bullets, and table-like structure. "
                        "Translate retrieved passages, explanations, tasks, education text, skills, "
                        "and generated report prose naturally. Do not add explanations. Output only Turkish."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Context: {context}\n\nText:\n{text}",
                },
            ],
            temperature=0.0,
        )
    except Exception:
        return text

    translated = (response.choices[0].message.content or "").strip()
    return translated or text


def _load_dotenv_if_available() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(dotenv_path=ENV_PATH)
