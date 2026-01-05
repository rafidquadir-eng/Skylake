"""
Multi-Omics Data Lake Harvester for Trained Immunity / Kidney Transplant Research

A comprehensive GCP Functions Framework app that harvests, processes, and curates
datasets from multiple biomedical repositories for a unified multi-omics datalake.

Supports all major data modalities:
- Transcriptomics (Bulk RNA, scRNA, Spatial, miRNA, Long-read, eccDNA/circRNA)
- Genomics (WGS, WES, HLA, SNP/GWAS, CNV)
- Epigenomics (ChIP-seq, ATAC-seq, Methylation, Hi-C)
- Proteomics & Signaling (Shotgun/DIA, Phospho, Cytokines, Complexome, Glycomics)
- Metabolism (Metabolomics, Lipidomics, Fluxomics)
- Immunophenotyping (Flow, CyTOF, Repertoire, Multiplex IHC, Functional)
- Microbiome (16S, Metagenomics)
- Integrated/Clinical (EHR, CITE-seq, Multiome, Perturb-seq)

Author: Cytoseeker Team
License: Research Use Only
"""

from __future__ import annotations

import functions_framework
import json
import os
import re
import time
import logging
import hashlib
import traceback
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Set, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import wraps
import threading

# External dependencies
import requests
from Bio import Entrez
from google.cloud import storage
from google import genai
from google.genai import types
from anthropic import AnthropicVertex, RateLimitError as AnthropicRateLimitError, APIError as AnthropicAPIError

# --- LOGGING CONFIGURATION ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("datalake_harvester")

# --- CONFIGURATION ---
PROJECT_ID = os.environ.get("GCP_PROJECT") or os.environ.get("GOOGLE_CLOUD_PROJECT")
BUCKET_NAME = os.environ.get("BUCKET_NAME", "cytoseeker-datalake")
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "5"))
MAX_RUNTIME_SECONDS = int(os.environ.get("MAX_RUNTIME_SECONDS", "3000"))  # 50 mins
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", "4"))

# API Keys and credentials
NCBI_API_KEY = os.environ.get("NCBI_API_KEY")
NCBI_EMAIL = os.environ.get("NCBI_EMAIL", "cytoseeker@research.local")

# --- CURATION PROVIDER CONFIGURATION ---
# Supported providers: "gemini" (recommended for high quota), "claude"
CURATION_PROVIDER = os.environ.get("CURATION_PROVIDER", "gemini").lower()

# Gemini/Vertex AI settings (recommended - high quota available)
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
VERTEX_LOCATION = os.environ.get("VERTEX_LOCATION", "us-central1")

# Claude via Vertex AI Model Garden (alternative - requires quota approval)
CLAUDE_REGION = os.environ.get("CLAUDE_REGION", "us-east5")
CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-haiku-4-5@20251001")

# Curation parallelism and rate limiting
# With 10B tokens/day quota, can run many parallel workers
CURATION_WORKERS = int(os.environ.get("CURATION_WORKERS", "8"))
CURATION_DELAY_SECONDS = float(os.environ.get("CURATION_DELAY_SECONDS", "0.1"))  # Minimal delay with high quota
# Maximum items to curate per invocation (0 = unlimited, constrained only by runtime)
MAX_CURATIONS_PER_RUN = int(os.environ.get("MAX_CURATIONS_PER_RUN", "1000"))

# Configure Entrez
Entrez.email = NCBI_EMAIL
if NCBI_API_KEY:
    Entrez.api_key = NCBI_API_KEY


# --- DATA CLASSES ---

class Modality(str, Enum):
    """Enumeration of all supported data modalities."""
    # Transcriptomics
    BULK_RNA = "bulk_rna"
    SCRNA = "scrna"
    SPATIAL = "spatial"
    MIRNA = "mirna"
    LONGREAD = "longread"
    ECCDNA = "eccdna"
    # Genomics
    WGS = "wgs"
    WES = "wes"
    HLA = "hla"
    SNP = "snp"
    CNV = "cnv"
    # Epigenomics
    CHIPSEQ = "chipseq"
    ATAC = "atac"
    METHYLATION = "methylation"
    HIC = "hic"
    # Proteomics & Signaling
    PROTEOMICS = "proteomics"
    PHOSPHO = "phospho"
    CYTOKINE = "cytokine"
    COMPLEXOME = "complexome"
    GLYCOMICS = "glycomics"
    # Metabolism
    METABOLOMICS = "metabolomics"
    LIPIDOMICS = "lipidomics"
    FLUXOMICS = "fluxomics"
    # Immunophenotyping
    FLOW = "flow"
    CYTOF = "cytof"
    REPERTOIRE = "repertoire"
    MIHC = "mihc"
    FUNCTIONAL = "functional"
    # Microbiome
    MICROBIOME_16S = "microbiome_16s"
    METAGENOMICS = "metagenomics"
    # Integrated/Clinical
    EHR = "ehr"
    CITESEQ = "citeseq"
    MULTIOME = "multiome"
    PERTURB = "perturb"


@dataclass
class HarvestResult:
    """Result from a harvesting operation."""
    source: str
    modality: str
    accession: str
    title: str
    summary: str
    organism: str
    sample_count: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    raw_data: Optional[Dict[str, Any]] = None
    curated: bool = False
    graph_functions: List[str] = field(default_factory=list)
    error: Optional[str] = None
    harvested_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class ProcessingState:
    """Tracks processing state for stateful resumption."""
    source: str
    last_processed_id: Optional[str] = None
    total_processed: int = 0
    total_errors: int = 0
    batch_number: int = 0
    started_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    completed: bool = False


# --- THE MEGA SCHEMA ---
MODALITY_SCHEMA = """
Strictly map data to these specific graph nodes. If a study covers multiple, list all relevant functions.

1. TRANSCRIPTOMICS (RNA):
   - build_bulk_rna_graph (Standard RNA-seq)
   - build_scrna_graph (Single-cell / Nucleus RNA-seq)
   - build_spatial_graph (Visium, GeoMx, MERFISH, Xenium)
   - build_mirna_graph (Small RNA / miRNA-seq)
   - build_longread_graph (Iso-seq, Nanopore Direct RNA)
   - build_eccdna_graph (eccDNA / circRNA)

2. GENOMICS (DNA):
   - build_wgs_graph (Whole Genome Sequencing)
   - build_wes_graph (Whole Exome Sequencing)
   - build_hla_graph (HLA Typing / Donor-Recipient Mismatch)
   - build_snp_graph (Genotyping Arrays / GWAS)
   - build_cnv_graph (Copy Number Variation)

3. EPIGENOMICS:
   - build_chipseq_graph (Histone marks / ChIP-seq)
   - build_atac_graph (Chromatin Accessibility / ATAC-seq / CUT&Tag / CUT&RUN)
   - build_methylation_graph (Bisulfite / EPIC Array / RRBS)
   - build_hic_graph (3D Chromatin / Hi-C / Micro-C)

4. PROTEOMICS & SIGNALING:
   - build_proteomics_graph (Shotgun / DIA / TMT / iTRAQ / Mass Spec)
   - build_phospho_graph (Phosphoproteomics / Kinase Signaling)
   - build_cytokine_graph (Olink / Luminex / ELISA / MSD)
   - build_complexome_graph (Interactome / Protein Complex / Co-IP)
   - build_glycomics_graph (Glycans / Glycosylation / Lectin Arrays)

5. METABOLISM:
   - build_metabolomics_graph (Targeted / Untargeted Metabolomics)
   - build_lipidomics_graph (Lipids / Lipid Mediators / Eicosanoids)
   - build_fluxomics_graph (Isotope Tracing / 13C Flux / Seahorse)

6. IMMUNOPHENOTYPING (CELLULAR):
   - build_flow_graph (Flow Cytometry / FACS / Spectral Flow)
   - build_cytof_graph (Mass Cytometry / CyTOF / Helios)
   - build_repertoire_graph (TCR-seq / BCR-seq / VDJ Repertoire)
   - build_mihc_graph (Multiplex IHC / IF / CODEX / MIBI / IMC / Phenocycler)
   - build_functional_graph (Functional Assays / Phospho-flow / Cytokine Secretion / ELISpot)

7. MICROBIOME:
   - build_microbiome_graph (16S rRNA / ITS Sequencing)
   - build_metagenomics_graph (Shotgun Metagenomics / Metatranscriptomics)

8. INTEGRATED / CLINICAL:
   - build_ehr_graph (Clinical Outcomes / Lab Values / Demographics / Survival)
   - build_citeseq_graph (CITE-seq / REAP-seq / DOGMA-seq / TotalSeq)
   - build_multiome_graph (scRNA + ATAC / 10x Multiome)
   - build_perturb_graph (CRISPR Screens / Perturb-seq / CROP-seq / TAP-seq)
"""

# --- DISEASE QUERIES FOR TRAINED IMMUNITY / KIDNEY TRANSPLANT ---
DISEASE_QUERIES = {
    "trained_immunity": [
        "trained immunity",
        "innate immune memory",
        "BCG vaccination immunity",
        "epigenetic reprogramming monocytes",
        "trained monocytes",
        "beta-glucan training",
        "metabolic rewiring immunity",
    ],
    "kidney_transplant": [
        "kidney transplant",
        "renal transplant",
        "kidney allograft",
        "transplant rejection",
        "kidney graft survival",
        "renal allograft dysfunction",
        "kidney transplant tolerance",
        "donor-specific antibodies kidney",
        "ischemia reperfusion kidney",
        "delayed graft function",
    ],
    "immunity_general": [
        "immune response",
        "inflammatory response",
        "cytokine storm",
        "immunomodulation",
        "innate immunity",
        "adaptive immunity",
    ],
}


# --- MODALITY-SPECIFIC QUERY TERMS ---
MODALITY_QUERIES = {
    # Transcriptomics
    Modality.BULK_RNA: ["RNA-seq", "transcriptome", "gene expression", "mRNA sequencing"],
    Modality.SCRNA: ["single-cell RNA", "scRNA-seq", "snRNA-seq", "single-nucleus RNA", "10x Genomics"],
    Modality.SPATIAL: ["spatial transcriptomics", "Visium", "GeoMx", "MERFISH", "Xenium", "Slide-seq", "FISH"],
    Modality.MIRNA: ["miRNA", "microRNA", "small RNA", "miRNA-seq"],
    Modality.LONGREAD: ["long-read RNA", "Iso-seq", "Nanopore RNA", "direct RNA sequencing", "PacBio RNA"],
    Modality.ECCDNA: ["eccDNA", "circRNA", "circular RNA", "extrachromosomal DNA"],
    # Genomics
    Modality.WGS: ["whole genome sequencing", "WGS", "genome sequencing"],
    Modality.WES: ["whole exome sequencing", "WES", "exome sequencing", "exome capture"],
    Modality.HLA: ["HLA typing", "HLA genotyping", "MHC typing", "donor recipient matching"],
    Modality.SNP: ["SNP array", "genotyping array", "GWAS", "genome-wide association"],
    Modality.CNV: ["copy number variation", "CNV", "copy number array", "aCGH"],
    # Epigenomics
    Modality.CHIPSEQ: ["ChIP-seq", "chromatin immunoprecipitation", "histone modification", "ChIP-exo"],
    Modality.ATAC: ["ATAC-seq", "chromatin accessibility", "CUT&Tag", "CUT&RUN", "DNase-seq", "FAIRE-seq"],
    Modality.METHYLATION: ["DNA methylation", "bisulfite sequencing", "WGBS", "RRBS", "EPIC array", "450K array"],
    Modality.HIC: ["Hi-C", "chromatin conformation", "3D chromatin", "Micro-C", "HiChIP", "PLAC-seq"],
    # Proteomics & Signaling
    Modality.PROTEOMICS: ["proteomics", "mass spectrometry", "LC-MS/MS", "TMT", "iTRAQ", "DIA", "SILAC"],
    Modality.PHOSPHO: ["phosphoproteomics", "phosphorylation", "kinase activity", "phospho-enrichment"],
    Modality.CYTOKINE: ["cytokine", "Olink", "Luminex", "ELISA", "MSD", "cytokine multiplex", "chemokine"],
    Modality.COMPLEXOME: ["protein complex", "interactome", "co-IP", "AP-MS", "BioID", "proximity ligation"],
    Modality.GLYCOMICS: ["glycomics", "glycan", "glycosylation", "lectin array", "glycoproteomics"],
    # Metabolism
    Modality.METABOLOMICS: ["metabolomics", "metabolome", "LC-MS metabolite", "GC-MS metabolite", "NMR metabolomics"],
    Modality.LIPIDOMICS: ["lipidomics", "lipid profiling", "eicosanoids", "lipid mediators", "sphingolipids"],
    Modality.FLUXOMICS: ["fluxomics", "metabolic flux", "13C tracing", "isotope tracing", "Seahorse", "ECAR", "OCR"],
    # Immunophenotyping
    Modality.FLOW: ["flow cytometry", "FACS", "spectral flow", "cell sorting"],
    Modality.CYTOF: ["CyTOF", "mass cytometry", "Helios", "metal-tagged antibodies"],
    Modality.REPERTOIRE: ["TCR sequencing", "BCR sequencing", "VDJ repertoire", "immune repertoire", "TCR-seq", "BCR-seq"],
    Modality.MIHC: ["multiplex IHC", "multiplex IF", "CODEX", "MIBI", "IMC", "Phenocycler", "imaging mass cytometry"],
    Modality.FUNCTIONAL: ["phospho-flow", "functional assay", "cytokine secretion", "ELISpot", "degranulation", "killing assay"],
    # Microbiome
    Modality.MICROBIOME_16S: ["16S rRNA", "16S sequencing", "microbiome", "gut microbiota", "ITS sequencing"],
    Modality.METAGENOMICS: ["metagenomics", "shotgun metagenomics", "metatranscriptomics", "viral metagenomics"],
    # Integrated/Clinical
    Modality.EHR: ["clinical outcome", "patient data", "electronic health record", "clinical trial", "survival data"],
    Modality.CITESEQ: ["CITE-seq", "REAP-seq", "DOGMA-seq", "TotalSeq", "surface protein single-cell"],
    Modality.MULTIOME: ["Multiome", "scRNA ATAC", "joint profiling", "10x Multiome", "SHARE-seq"],
    Modality.PERTURB: ["Perturb-seq", "CRISPR screen", "CROP-seq", "TAP-seq", "Mosaic-seq", "pooled CRISPR"],
}


# --- FILE PATTERNS BY MODALITY (for data matrix discovery) ---
# Priority order: first patterns are preferred
# Note: .gz compression is handled automatically in get_file_priority()
MODALITY_FILE_PATTERNS = {
    # Transcriptomics
    Modality.BULK_RNA: {
        "priority": [
            # TAR archives with raw data (highest priority)
            "*_RAW.tar", "*_raw.tar",
            # Normalized expression matrices
            "*_CPM*", "*_cpm*", "*CPM*.csv", "*CPM*.txt",
            "*_TPM*", "*_tpm*", "*TPM*.csv", "*TPM*.txt",
            "*_FPKM*", "*_fpkm*", "*FPKM*.csv", "*FPKM*.txt",
            "*_RPKM*", "*_rpkm*",
            # Raw counts
            "*_counts*", "*_count_matrix*", "*counts*.txt", "*counts*.csv",
            "*_raw_counts*", "*raw*counts*",
            "*expression*matrix*", "*expression*.csv", "*expression*.txt",
            # HDF5 formats
            "*.h5", "*.h5ad",
            # Generic data files
            "*_data.csv", "*_data.txt", "*_normalized*",
        ],
        "extensions": [".txt", ".csv", ".tsv", ".h5", ".h5ad", ".tar"],
        "exclude": ["*README*", "*sample*info*", "*phenotype*", "*filelist*"],
    },
    Modality.SCRNA: {
        "priority": [
            # TAR archives (contain matrix files inside)
            "*_RAW.tar", "*_raw.tar",
            # AnnData / HDF5 (best - complete datasets)
            "*.h5ad", "*.h5",
            # 10x sparse matrix format
            "matrix.mtx*", "*matrix.mtx*",
            "barcodes.tsv*", "barcodes*.tsv*",
            "features.tsv*", "genes.tsv*", "features*.tsv*",
            # 10x processed outputs
            "*filtered_feature_bc_matrix*", "*_filtered_*",
            "*raw_feature_bc_matrix*",
            # Count matrices
            "*counts*", "*_count_matrix*", "*expression*",
            # Loom format
            "*.loom",
            # Seurat objects
            "*.rds",
        ],
        "extensions": [".h5ad", ".h5", ".mtx", ".loom", ".rds", ".tar"],
        "exclude": ["*filelist*"],
    },
    Modality.SPATIAL: {
        "priority": [
            "*_RAW.tar", "*_raw.tar",
            "*.h5ad", "*.h5",
            "matrix.mtx*", "*matrix.mtx*",
            "tissue_positions*", "*positions*.csv",
            "scalefactors*",
            "*spatial*", "*visium*", "*xenium*", "*cosmx*",
            "barcodes.tsv*", "features.tsv*",
            "*counts*", "*expression*",
        ],
        "extensions": [".h5ad", ".h5", ".mtx", ".csv", ".json", ".tar"],
        "exclude": ["*filelist*"],
    },
    Modality.MIRNA: {
        "priority": [
            "*_RAW.tar",
            "*miRNA*counts*", "*miRNA*expression*", "*mirna*",
            "*_counts*", "*_TPM*", "*_CPM*",
            "*small_rna*counts*", "*smallrna*",
            "*expression*",
        ],
        "extensions": [".txt", ".csv", ".tsv", ".tar"],
        "exclude": ["*filelist*"],
    },
    Modality.LONGREAD: {
        "priority": [
            "*_RAW.tar",
            "*isoform*counts*", "*transcript*counts*",
            "*_counts*", "*_TPM*",
            "*.h5", "*.h5ad",
        ],
        "extensions": [".txt", ".csv", ".h5", ".h5ad", ".tar"],
        "exclude": ["*filelist*"],
    },
    # Genomics
    Modality.WGS: {
        "priority": [
            "*.vcf.gz", "*.vcf",  # Variant calls
            "*.maf", "*.maf.gz",  # Mutation annotation
            "*_mutations.txt", "*_variants.txt",
        ],
        "extensions": [".vcf", ".vcf.gz", ".maf", ".maf.gz"],
        "exclude": [],
    },
    Modality.WES: {
        "priority": [
            "*.vcf.gz", "*.vcf",
            "*.maf", "*.maf.gz",
            "*_mutations.txt", "*_variants.txt",
        ],
        "extensions": [".vcf", ".vcf.gz", ".maf", ".maf.gz"],
        "exclude": [],
    },
    Modality.HLA: {
        "priority": [
            "*hla*genotype*", "*HLA*typing*",
            "*_hla.csv", "*_hla.txt",
            "*allele*", "*mismatch*",
        ],
        "extensions": [".csv", ".txt", ".tsv"],
        "exclude": [],
    },
    Modality.SNP: {
        "priority": [
            "*.vcf.gz", "*.vcf",
            "*_genotypes.txt", "*_snp*.txt",
            "*gwas*", "*association*",
        ],
        "extensions": [".vcf", ".vcf.gz", ".txt", ".csv"],
        "exclude": [],
    },
    Modality.CNV: {
        "priority": [
            "*.seg", "*_segments.txt",  # Segment files
            "*_cnv*.txt", "*copy_number*",
            "*.cns", "*.cnr",  # CNVkit output
        ],
        "extensions": [".seg", ".txt", ".csv", ".cns", ".cnr"],
        "exclude": [],
    },
    # Epigenomics
    Modality.ATAC: {
        "priority": [
            "*peaks*count*matrix*", "*peak*counts*",  # Peak count matrix (best)
            "*.narrowPeak", "*.broadPeak",  # Peak calls
            "*.bigWig", "*.bw",  # Signal tracks
            "*.bed", "*.bed.gz",
        ],
        "extensions": [".txt", ".narrowPeak", ".broadPeak", ".bw", ".bigWig", ".bed"],
        "exclude": [],
    },
    Modality.CHIPSEQ: {
        "priority": [
            "*peaks*count*matrix*", "*peak*counts*",
            "*.narrowPeak", "*.broadPeak",
            "*.bigWig", "*.bw",
            "*.bed", "*.bed.gz",
        ],
        "extensions": [".txt", ".narrowPeak", ".broadPeak", ".bw", ".bigWig", ".bed"],
        "exclude": [],
    },
    Modality.METHYLATION: {
        "priority": [
            "*beta*values*", "*_beta.txt",  # Beta values (best)
            "*methylation*matrix*",
            "*.idat",  # Raw intensity
            "*_M_values*",
        ],
        "extensions": [".txt", ".csv", ".idat"],
        "exclude": [],
    },
    Modality.HIC: {
        "priority": [
            "*.hic",  # Juicebox format
            "*.cool", "*.mcool",  # Cooler format
            "*contact*matrix*",
            "*.pairs", "*.pairs.gz",
        ],
        "extensions": [".hic", ".cool", ".mcool", ".pairs"],
        "exclude": [],
    },
    # Proteomics
    Modality.PROTEOMICS: {
        "priority": [
            "proteinGroups.txt",  # MaxQuant
            "report.tsv", "report.txt",  # DIA-NN
            "*protein*intensity*", "*_LFQ*",
            "*abundance*matrix*",
        ],
        "extensions": [".txt", ".tsv", ".csv"],
        "exclude": ["*peptides*", "*evidence*"],
    },
    Modality.PHOSPHO: {
        "priority": [
            "*Phospho*STY*Sites*",  # MaxQuant phospho
            "*phospho*sites*",
            "*phosphorylation*",
            "*kinase*activity*",
        ],
        "extensions": [".txt", ".tsv", ".csv"],
        "exclude": [],
    },
    Modality.CYTOKINE: {
        "priority": [
            "*_NPX*.csv", "*_NPX*.txt",  # Olink
            "*Luminex*", "*multiplex*",
            "*cytokine*concentration*",
            "*ELISA*results*",
        ],
        "extensions": [".csv", ".txt", ".xlsx"],
        "exclude": [],
    },
    # Immunophenotyping
    Modality.FLOW: {
        "priority": [
            "*.fcs",  # Raw flow data
            "*population*counts*", "*cell*counts*",
            "*gating*stats*", "*frequencies*",
        ],
        "extensions": [".fcs", ".csv", ".txt"],
        "exclude": [],
    },
    Modality.CYTOF: {
        "priority": [
            "*.fcs",  # CyTOF also uses FCS
            "*population*counts*", "*cell*counts*",
            "*frequencies*",
        ],
        "extensions": [".fcs", ".csv", ".txt"],
        "exclude": [],
    },
    Modality.REPERTOIRE: {
        "priority": [
            "filtered_contig_annotations.csv",  # 10x
            "*clonotypes*", "*_tcr.csv", "*_bcr.csv",
            "*airr*.tsv",  # AIRR format
            "*vdj*annotations*",
        ],
        "extensions": [".csv", ".tsv", ".txt"],
        "exclude": [],
    },
    # Microbiome / Metabolism
    Modality.MICROBIOME_16S: {
        "priority": [
            "*otu*table*", "*asv*table*",  # OTU/ASV tables
            "*.biom",  # BIOM format
            "*taxonomy*",
            "*abundance*matrix*",
        ],
        "extensions": [".biom", ".csv", ".tsv", ".txt"],
        "exclude": [],
    },
    Modality.METAGENOMICS: {
        "priority": [
            "*abundance*matrix*", "*gene*counts*",
            "*kraken*", "*metaphlan*",
            "*.biom",
        ],
        "extensions": [".biom", ".csv", ".tsv", ".txt"],
        "exclude": [],
    },
    Modality.METABOLOMICS: {
        "priority": [
            "*peak*table*", "*intensity*matrix*",
            "*metabolite*abundance*",
            "*_mz_*.csv",
        ],
        "extensions": [".csv", ".txt", ".tsv", ".xlsx"],
        "exclude": [],
    },
    Modality.LIPIDOMICS: {
        "priority": [
            "*lipid*intensity*", "*lipid*abundance*",
            "*peak*table*",
        ],
        "extensions": [".csv", ".txt", ".tsv"],
        "exclude": [],
    },
    # Multi-omics
    Modality.CITESEQ: {
        "priority": [
            "*.h5ad", "*.h5",
            "*protein*counts*", "*adt*counts*",
            "matrix.mtx.gz",
        ],
        "extensions": [".h5ad", ".h5", ".mtx", ".mtx.gz"],
        "exclude": [],
    },
    Modality.MULTIOME: {
        "priority": [
            "*.h5ad", "*.h5",
            "*rna*counts*", "*atac*counts*",
            "matrix.mtx.gz",
        ],
        "extensions": [".h5ad", ".h5", ".mtx", ".mtx.gz"],
        "exclude": [],
    },
    Modality.PERTURB: {
        "priority": [
            "*.h5ad", "*.h5",
            "*perturbation*matrix*", "*guide*counts*",
            "matrix.mtx.gz",
        ],
        "extensions": [".h5ad", ".h5", ".mtx", ".mtx.gz"],
        "exclude": [],
    },
}

# Default patterns for modalities not explicitly defined
DEFAULT_FILE_PATTERNS = {
    "priority": [
        "*_RAW.tar", "*_raw.tar",  # TAR archives
        "*.h5ad", "*.h5",  # HDF5 formats
        "*_counts*", "*_count*", "*counts*",
        "*matrix*", "*expression*",
        "*_CPM*", "*_TPM*", "*_FPKM*",
        "*_data.csv", "*_data.txt",
    ],
    "extensions": [".txt", ".csv", ".tsv", ".h5", ".h5ad", ".tar"],
    "exclude": ["*README*", "*sample*info*", "*filelist*", "*.html"],
}


# --- GLOBAL NCBI RATE LIMITER ---
# This ensures thread-safe rate limiting across all parallel harvesters

class NCBIRateLimiter:
    """
    Thread-safe global rate limiter for NCBI API calls.
    
    NCBI allows:
    - 3 requests/second without API key
    - 10 requests/second with API key
    
    We use a conservative 2.5 req/sec to avoid 429 errors.
    """
    
    def __init__(self, requests_per_second: float = 2.5):
        self.min_interval = 1.0 / requests_per_second
        self.lock = threading.Lock()
        self.last_request_time = 0.0
    
    def acquire(self):
        """Block until it's safe to make a request."""
        with self.lock:
            now = time.time()
            elapsed = now - self.last_request_time
            if elapsed < self.min_interval:
                sleep_time = self.min_interval - elapsed
                time.sleep(sleep_time)
            self.last_request_time = time.time()


# Single global instance - shared across ALL threads
_NCBI_RATE_LIMITER = NCBIRateLimiter(requests_per_second=2.5)


# --- UTILITY FUNCTIONS ---

def clean_metadata(obj: Any) -> Any:
    """
    Recursively convert Biopython DictionaryElement and other non-serializable 
    objects to plain Python types for JSON serialization.
    
    Fixes the 'DictionaryElement.__init__() missing arguments' crash.
    
    This handles:
    - Bio.Entrez.Parser.DictionaryElement -> dict
    - Bio.Entrez.Parser.ListElement -> list  
    - Bio.Entrez.Parser.StringElement -> str
    - Other objects -> str representation
    """
    # Handle None
    if obj is None:
        return None
    
    # Handle basic types (already serializable)
    if isinstance(obj, (bool, int, float, str)):
        return obj
    
    # Handle dict-like objects (including DictionaryElement)
    if isinstance(obj, dict):
        return {str(k): clean_metadata(v) for k, v in obj.items()}
    
    # Handle list-like objects (including ListElement)
    if isinstance(obj, (list, tuple)):
        return [clean_metadata(item) for item in obj]
    
    # Handle objects with .items() method (dict-like)
    if hasattr(obj, 'items'):
        try:
            return {str(k): clean_metadata(v) for k, v in obj.items()}
        except Exception:
            pass
    
    # Handle objects that are iterable but not string-like
    if hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes)):
        try:
            return [clean_metadata(item) for item in obj]
        except Exception:
            pass
    
    # Fallback: convert to string
    try:
        return str(obj)
    except Exception:
        return repr(obj)


def rate_limit(min_interval: float = 0.35):
    """Decorator for rate limiting API calls (for non-NCBI APIs)."""
    last_call = [0.0]
    lock = threading.Lock()
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            with lock:
                elapsed = time.time() - last_call[0]
                if elapsed < min_interval:
                    time.sleep(min_interval - elapsed)
                last_call[0] = time.time()
            return func(*args, **kwargs)
        return wrapper
    return decorator


def retry_with_backoff(max_retries: int = 3, base_delay: float = 2.0, retry_on_429: bool = True):
    """
    Decorator for retry with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Base delay in seconds (doubles each attempt)
        retry_on_429: If True, applies extra-long backoff for rate limit (429) errors
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    error_str = str(e)
                    
                    # Apply longer backoff for rate limit errors
                    is_rate_limit = "429" in error_str or "RESOURCE_EXHAUSTED" in error_str
                    if is_rate_limit and retry_on_429:
                        # Much longer backoff for quota errors (30s, 60s, 120s)
                        delay = 30 * (2 ** attempt)
                        logger.warning(
                            f"Rate limit hit (attempt {attempt + 1}/{max_retries}): {e}. "
                            f"Backing off for {delay}s..."
                        )
                    else:
                        delay = base_delay * (2 ** attempt)
                        logger.warning(f"Attempt {attempt + 1}/{max_retries} failed: {e}. Retrying in {delay}s...")
                    
                    time.sleep(delay)
            raise last_exception
        return wrapper
    return decorator


def get_storage_client():
    """Get GCS storage client."""
    try:
        return storage.Client(project=PROJECT_ID)
    except Exception as e:
        logger.warning(f"Could not create GCS client: {e}")
        return None


def load_state(source: str) -> Optional[ProcessingState]:
    """Load processing state from GCS."""
    client = get_storage_client()
    if not client:
        return None
    try:
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(f"harvest_state/{source}_state.json")
        if blob.exists():
            data = json.loads(blob.download_as_text())
            return ProcessingState(**data)
    except Exception as e:
        logger.warning(f"Could not load state for {source}: {e}")
    return None


def save_state(state: ProcessingState) -> bool:
    """Save processing state to GCS."""
    client = get_storage_client()
    if not client:
        return False
    try:
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(f"harvest_state/{state.source}_state.json")
        state.updated_at = datetime.utcnow().isoformat()
        blob.upload_from_string(json.dumps(asdict(state)), content_type="application/json")
        return True
    except Exception as e:
        logger.error(f"Could not save state for {state.source}: {e}")
        return False


def save_result(result: HarvestResult) -> bool:
    """Save harvest result to GCS with JSON-safe serialization."""
    client = get_storage_client()
    if not client:
        logger.info(f"[LOCAL MODE] Would save result: {result.source}/{result.accession}")
        return True
    try:
        bucket = client.bucket(BUCKET_NAME)
        date_prefix = datetime.utcnow().strftime("%Y/%m/%d")
        blob_path = f"harvest_results/{result.source}/{date_prefix}/{result.accession}.json"
        blob = bucket.blob(blob_path)
        
        # Convert to dict and make JSON-serializable
        # This handles Biopython DictionaryElement and other non-serializable objects
        result_dict = asdict(result)
        safe_dict = clean_metadata(result_dict)
        
        blob.upload_from_string(json.dumps(safe_dict), content_type="application/json")
        logger.info(f"Saved result: gs://{BUCKET_NAME}/{blob_path}")
        return True
    except Exception as e:
        logger.error(f"Could not save result {result.accession}: {e}")
        logger.debug(f"Serialization error details: {traceback.format_exc()}")
        return False


# --- FILE DISCOVERY AND DOWNLOAD ---

@dataclass
class DataFile:
    """Represents a downloadable data file."""
    filename: str
    url: str
    size_bytes: Optional[int] = None
    file_type: str = ""  # e.g., "counts_matrix", "h5ad", "vcf"
    modality_match: bool = False  # True if matches modality patterns
    priority_score: int = 0  # Higher = better match
    downloaded: bool = False
    gcs_path: Optional[str] = None


def match_file_pattern(filename: str, pattern: str) -> bool:
    """
    Check if filename matches a glob-like pattern.
    Supports * as wildcard.
    """
    import fnmatch
    return fnmatch.fnmatch(filename.lower(), pattern.lower())


def get_file_priority(filename: str, modality: str) -> Tuple[bool, int]:
    """
    Determine if a file matches the modality patterns and its priority.
    
    Handles compressed files (.gz, .zip, .bz2) by checking the base filename.
    TAR archives are high priority as they typically contain raw data matrices.
    
    Returns:
        (matches_modality, priority_score)
    """
    filename_lower = filename.lower()
    
    # Strip compression extensions to get base filename for pattern matching
    base_filename = filename_lower
    is_compressed = False
    for comp_ext in ['.gz', '.zip', '.bz2', '.xz', '.zst']:
        if base_filename.endswith(comp_ext):
            base_filename = base_filename[:-len(comp_ext)]
            is_compressed = True
            break
    
    # TAR archives are very high priority - they contain the actual data files
    # GEO typically packages raw data as GSExxxxx_RAW.tar
    if filename_lower.endswith('.tar') or '_raw.tar' in filename_lower:
        return True, 20  # Highest priority - contains matrix files
    
    # Check for common data file indicators (works across all modalities)
    universal_high_priority = [
        '_cpm', '_tpm', '_fpkm', '_rpkm',  # Normalized counts
        '_counts', '_count_matrix', '_expression',
        '_raw_counts', '_normalized',
        'matrix.mtx', 'counts.csv', 'counts.txt', 'counts.tsv',
    ]
    for indicator in universal_high_priority:
        if indicator in base_filename:
            return True, 15
    
    # Get patterns for this modality
    try:
        mod_enum = Modality(modality) if isinstance(modality, str) else modality
        patterns = MODALITY_FILE_PATTERNS.get(mod_enum, DEFAULT_FILE_PATTERNS)
    except (ValueError, KeyError):
        patterns = DEFAULT_FILE_PATTERNS
    
    # Check exclude patterns first
    for exclude in patterns.get("exclude", []):
        if match_file_pattern(filename_lower, exclude) or match_file_pattern(base_filename, exclude):
            return False, 0
    
    # Check priority patterns against both compressed and uncompressed names
    priority_list = patterns.get("priority", [])
    for i, pattern in enumerate(priority_list):
        if match_file_pattern(filename_lower, pattern) or match_file_pattern(base_filename, pattern):
            # Priority score: higher for earlier patterns
            return True, len(priority_list) - i + 5
    
    # Check if extension matches (check base extension for compressed files)
    extensions = patterns.get("extensions", [])
    
    # Get the actual file extension
    if '.' in base_filename:
        actual_ext = '.' + base_filename.split('.')[-1]
    else:
        actual_ext = ''
    
    for ext in extensions:
        ext_lower = ext.lower()
        if base_filename.endswith(ext_lower) or actual_ext == ext_lower:
            return True, 3 if is_compressed else 2  # Compressed files slightly higher
    
    # CSV/TSV/TXT files are generally useful even if not matching specific patterns
    generic_data_extensions = ['.csv', '.tsv', '.txt', '.xlsx']
    for ext in generic_data_extensions:
        if base_filename.endswith(ext):
            return True, 1  # Low priority but still download
    
    return False, 0


def _parse_geo_directory(url: str, base_url: str) -> List[DataFile]:
    """
    Parse a GEO FTP directory listing and extract files.
    
    Args:
        url: URL to fetch
        base_url: Base URL for constructing file URLs
    
    Returns:
        List of DataFile objects
    """
    files = []
    try:
        response = requests.get(url, timeout=30)
        
        if response.status_code == 200:
            content = response.text
            
            # Extract file links using regex
            import re
            file_pattern = re.compile(r'href="([^"]+\.[^"]+)"')
            matches = file_pattern.findall(content)
            
            for filename in matches:
                # Skip unwanted entries
                if filename.startswith("..") or filename.startswith("/"):
                    continue
                if filename.startswith("http://") or filename.startswith("https://"):
                    continue
                if filename in ["filelist.txt"]:
                    continue
                
                file_url = f"{base_url}{filename}"
                ext = filename.split(".")[-1].lower() if "." in filename else ""
                
                files.append(DataFile(
                    filename=filename,
                    url=file_url,
                    file_type=ext,
                ))
    except Exception as e:
        logger.debug(f"Error parsing GEO directory {url}: {e}")
    
    return files


@rate_limit(0.5)
def discover_geo_files(accession: str) -> List[DataFile]:
    """
    Discover all available files for a GEO dataset.
    
    Checks multiple locations:
    1. /suppl/ directory - Supplementary files (processed data)
    2. /matrix/ directory - Series matrix files (expression matrices)
    3. GEO download API - Direct file access
    
    Args:
        accession: GEO accession (e.g., "GSE12345")
    
    Returns:
        List of DataFile objects
    """
    files = []
    seen_filenames = set()  # Avoid duplicates
    
    if not accession.startswith("GSE"):
        logger.debug(f"Skipping non-GSE accession: {accession}")
        return files
    
    # Build FTP path structure
    # GSE12345 -> series/GSE12nnn/GSE12345/
    gse_num = accession[3:]  # Remove "GSE" prefix
    series_folder = f"GSE{gse_num[:-3]}nnn" if len(gse_num) > 3 else "GSE0nnn"
    
    ftp_base = "https://ftp.ncbi.nlm.nih.gov"
    base_path = f"{ftp_base}/geo/series/{series_folder}/{accession}"
    
    # 1. Check /suppl/ directory (supplementary files)
    suppl_url = f"{base_path}/suppl/"
    suppl_files = _parse_geo_directory(suppl_url, suppl_url)
    for f in suppl_files:
        if f.filename not in seen_filenames:
            seen_filenames.add(f.filename)
            files.append(f)
    
    # 2. Check /matrix/ directory (series matrix - expression data)
    matrix_url = f"{base_path}/matrix/"
    matrix_files = _parse_geo_directory(matrix_url, matrix_url)
    for f in matrix_files:
        if f.filename not in seen_filenames:
            seen_filenames.add(f.filename)
            # Series matrix files are high priority
            f.priority_score = 18
            files.append(f)
    
    # 3. Try direct series matrix URL (sometimes not in directory listing)
    series_matrix_name = f"{accession}_series_matrix.txt.gz"
    if series_matrix_name not in seen_filenames:
        series_matrix_url = f"{matrix_url}{series_matrix_name}"
        try:
            # Check if file exists with HEAD request
            head_resp = requests.head(series_matrix_url, timeout=10)
            if head_resp.status_code == 200:
                seen_filenames.add(series_matrix_name)
                files.append(DataFile(
                    filename=series_matrix_name,
                    url=series_matrix_url,
                    file_type="gz",
                    priority_score=18,
                ))
        except:
            pass
    
    # 4. Try GEO download API for RAW files bundle
    # https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSExxxxx&format=file
    raw_bundle_name = f"{accession}_RAW.tar"
    if raw_bundle_name not in seen_filenames:
        raw_bundle_url = f"https://www.ncbi.nlm.nih.gov/geo/download/?acc={accession}&format=file"
        try:
            # Check if RAW bundle exists
            head_resp = requests.head(raw_bundle_url, timeout=10, allow_redirects=True)
            if head_resp.status_code == 200:
                content_type = head_resp.headers.get("content-type", "")
                # GEO returns application/x-tar for actual tar files
                if "tar" in content_type or "octet" in content_type:
                    seen_filenames.add(raw_bundle_name)
                    files.append(DataFile(
                        filename=raw_bundle_name,
                        url=raw_bundle_url,
                        file_type="tar",
                        priority_score=20,  # High priority - contains raw data
                    ))
        except:
            pass
    
    # 5. Try to discover individual supplementary files via GEO soft file
    # Parse the SOFT file to find file names if we have very few files
    if len(files) < 2:
        try:
            soft_url = f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={accession}&targ=self&form=text&view=brief"
            soft_resp = requests.get(soft_url, timeout=15)
            if soft_resp.status_code == 200:
                soft_text = soft_resp.text
                
                # Look for supplementary file references
                import re
                # Pattern to match supplementary file URLs or names
                suppl_pattern = re.compile(r'!Series_supplementary_file\s*=\s*(.+)', re.IGNORECASE)
                for match in suppl_pattern.findall(soft_text):
                    match = match.strip()
                    if match.startswith("ftp://") or match.startswith("http"):
                        # Full URL provided
                        filename = match.split("/")[-1]
                        if filename and filename not in seen_filenames:
                            # Convert ftp:// to https://
                            file_url = match.replace("ftp://ftp.ncbi.nlm.nih.gov", "https://ftp.ncbi.nlm.nih.gov")
                            seen_filenames.add(filename)
                            ext = filename.split(".")[-1].lower() if "." in filename else ""
                            files.append(DataFile(
                                filename=filename,
                                url=file_url,
                                file_type=ext,
                            ))
        except Exception as e:
            logger.debug(f"Could not parse SOFT file for {accession}: {e}")
    
    if files:
        logger.info(f"Found {len(files)} files for GEO {accession} (suppl: {len(suppl_files)}, matrix: {len(matrix_files)})")
    else:
        logger.debug(f"No files found for GEO {accession}")
    
    return files


@rate_limit(1.0)
def discover_arrayexpress_files(accession: str) -> List[DataFile]:
    """
    Discover files for an ArrayExpress/BioStudies dataset.
    
    Uses multiple EBI API endpoints as the API structure varies.
    For E-GEOD accessions (GEO mirrors), redirects to GEO FTP.
    
    Args:
        accession: ArrayExpress accession (e.g., "E-MTAB-12345", "E-GEOD-12345")
    
    Returns:
        List of DataFile objects
    """
    files = []
    
    if not (accession.startswith("E-") or accession.startswith("S-")):
        return files
    
    # E-GEOD accessions are GEO mirrors - redirect to GEO FTP
    if accession.startswith("E-GEOD-"):
        gse_num = accession.replace("E-GEOD-", "")
        gse_accession = f"GSE{gse_num}"
        logger.debug(f"Redirecting E-GEOD to GEO: {accession} -> {gse_accession}")
        return discover_geo_files(gse_accession)
    
    # Try multiple BioStudies API endpoints
    api_endpoints = [
        # New BioStudies API
        f"https://www.ebi.ac.uk/biostudies/api/v1/studies/{accession}",
        # Files endpoint directly
        f"https://www.ebi.ac.uk/biostudies/api/v1/studies/{accession}/files",
        # ArrayExpress specific endpoint
        f"https://www.ebi.ac.uk/arrayexpress/json/v3/experiments/{accession}/files",
    ]
    
    for api_url in api_endpoints:
        try:
            response = requests.get(api_url, timeout=30, headers={"Accept": "application/json"})
            
            if response.status_code != 200:
                continue
            
            data = response.json()
            
            # Handle different response structures
            file_list = []
            
            # BioStudies v1 structure
            if "section" in data:
                section = data.get("section", {})
                # Files can be in section.files or nested in subsections
                file_list.extend(section.get("files", []))
                for subsection in section.get("subsections", []):
                    if isinstance(subsection, dict):
                        file_list.extend(subsection.get("files", []))
            
            # Direct files array
            elif isinstance(data, list):
                file_list = data
            elif "files" in data:
                file_list = data.get("files", [])
            
            # Process file list
            for file_info in file_list:
                if isinstance(file_info, dict):
                    # Try different field names for filename
                    filename = (
                        file_info.get("path") or 
                        file_info.get("name") or 
                        file_info.get("fileName") or
                        ""
                    )
                    size = file_info.get("size", 0)
                    
                    if filename and not filename.startswith("/"):
                        # Build download URL
                        file_url = f"https://www.ebi.ac.uk/biostudies/files/{accession}/{filename}"
                        ext = filename.split(".")[-1].lower() if "." in filename else ""
                        
                        data_file = DataFile(
                            filename=filename,
                            url=file_url,
                            size_bytes=size,
                            file_type=ext,
                        )
                        files.append(data_file)
                elif isinstance(file_info, str) and file_info:
                    # Sometimes files are just strings (filenames)
                    filename = file_info
                    file_url = f"https://www.ebi.ac.uk/biostudies/files/{accession}/{filename}"
                    ext = filename.split(".")[-1].lower() if "." in filename else ""
                    
                    data_file = DataFile(
                        filename=filename,
                        url=file_url,
                        file_type=ext,
                    )
                    files.append(data_file)
            
            if files:
                logger.info(f"Found {len(files)} files for ArrayExpress {accession}")
                break  # Stop trying other endpoints
                
        except Exception as e:
            logger.debug(f"ArrayExpress API {api_url} failed for {accession}: {e}")
            continue
    
    # If no files found via API, try direct FTP listing
    if not files:
        ftp_url = f"https://ftp.ebi.ac.uk/biostudies/fire/E-MTAB/{accession}/"
        try:
            response = requests.get(ftp_url, timeout=15)
            if response.status_code == 200:
                import re
                file_pattern = re.compile(r'href="([^"]+\.[^"]+)"')
                matches = file_pattern.findall(response.text)
                for filename in matches:
                    if not filename.startswith("..") and not filename.startswith("/"):
                        file_url = f"{ftp_url}{filename}"
                        ext = filename.split(".")[-1].lower() if "." in filename else ""
                        files.append(DataFile(
                            filename=filename,
                            url=file_url,
                            file_type=ext,
                        ))
                if files:
                    logger.info(f"Found {len(files)} files via FTP for {accession}")
        except:
            pass
    
    if not files:
        logger.debug(f"No files found for ArrayExpress {accession}")
    
    return files


@rate_limit(1.0)
def discover_pride_files(accession: str) -> List[DataFile]:
    """
    Discover files for a PRIDE proteomics dataset.
    
    Uses PRIDE Archive API.
    
    Args:
        accession: PRIDE accession (e.g., "PXD012345")
    
    Returns:
        List of DataFile objects
    """
    files = []
    
    if not accession.startswith("PXD"):
        return files
    
    # PRIDE API endpoint
    api_url = f"https://www.ebi.ac.uk/pride/ws/archive/v2/projects/{accession}/files"
    
    try:
        response = requests.get(api_url, timeout=30, headers={"Accept": "application/json"})
        
        if response.status_code == 200:
            data = response.json()
            
            for file_info in data:
                filename = file_info.get("fileName", "")
                download_link = file_info.get("downloadLink", "")
                size = file_info.get("fileSize", 0)
                file_type = file_info.get("fileType", "")
                
                if filename and download_link:
                    ext = filename.split(".")[-1].lower() if "." in filename else ""
                    
                    # Prioritize processed result files
                    priority = 0
                    if "result" in file_type.lower() or "protein" in filename.lower():
                        priority = 10
                    elif "search" in file_type.lower():
                        priority = 5
                    
                    data_file = DataFile(
                        filename=filename,
                        url=download_link,
                        size_bytes=size,
                        file_type=ext,
                        priority_score=priority,
                    )
                    files.append(data_file)
            
            logger.info(f"Found {len(files)} files for PRIDE {accession}")
        else:
            logger.debug(f"PRIDE API returned {response.status_code} for {accession}")
            
    except Exception as e:
        logger.warning(f"Error discovering PRIDE files for {accession}: {e}")
    
    return files


@rate_limit(1.0)
def discover_flowrepository_files(accession: str) -> List[DataFile]:
    """
    Discover FCS files from FlowRepository.
    
    FlowRepository hosts flow cytometry data in FCS format.
    
    Args:
        accession: FlowRepository ID (e.g., "FR-FCM-XXXX")
    
    Returns:
        List of DataFile objects
    """
    files = []
    
    if not accession.startswith("FR-FCM"):
        return files
    
    # FlowRepository API
    api_url = f"https://flowrepository.org/ajax/repo_dataset_files?id={accession}"
    
    try:
        response = requests.get(api_url, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            
            for file_info in data.get("files", []):
                filename = file_info.get("name", "")
                file_id = file_info.get("id", "")
                size = file_info.get("size", 0)
                
                if filename and file_id:
                    # Build download URL
                    file_url = f"https://flowrepository.org/lib/download_file/{file_id}"
                    ext = filename.split(".")[-1].lower() if "." in filename else ""
                    
                    data_file = DataFile(
                        filename=filename,
                        url=file_url,
                        size_bytes=size,
                        file_type=ext,
                        priority_score=10 if ext == "fcs" else 1,
                    )
                    files.append(data_file)
            
            logger.info(f"Found {len(files)} files for FlowRepository {accession}")
        else:
            logger.debug(f"FlowRepository API returned {response.status_code} for {accession}")
            
    except Exception as e:
        logger.warning(f"Error discovering FlowRepository files for {accession}: {e}")
    
    return files


@rate_limit(1.0)
def discover_immport_files(accession: str) -> List[DataFile]:
    """
    Discover files from ImmPort.
    
    ImmPort is an immunology data repository.
    Note: ImmPort requires authentication for most downloads.
    
    Args:
        accession: ImmPort study ID (e.g., "SDY1")
    
    Returns:
        List of DataFile objects with download info
    """
    files = []
    
    if not accession.startswith("SDY"):
        return files
    
    # ImmPort public data catalog
    api_url = f"https://www.immport.org/data/study/{accession}/files"
    
    try:
        response = requests.get(api_url, timeout=30, headers={"Accept": "application/json"})
        
        if response.status_code == 200:
            data = response.json()
            
            for file_info in data.get("files", data if isinstance(data, list) else []):
                filename = file_info.get("fileName", file_info.get("name", ""))
                file_path = file_info.get("path", "")
                size = file_info.get("fileSize", file_info.get("size", 0))
                
                if filename:
                    # ImmPort requires auth, so we store the reference
                    file_url = f"https://www.immport.org/data/study/{accession}/files/{file_path}"
                    ext = filename.split(".")[-1].lower() if "." in filename else ""
                    
                    # Prioritize flow cytometry and processed data
                    priority = 0
                    if ext == "fcs":
                        priority = 10
                    elif "population" in filename.lower() or "gating" in filename.lower():
                        priority = 8
                    elif ext in ["csv", "txt", "tsv"]:
                        priority = 5
                    
                    data_file = DataFile(
                        filename=filename,
                        url=file_url,
                        size_bytes=size,
                        file_type=ext,
                        priority_score=priority,
                    )
                    files.append(data_file)
            
            logger.info(f"Found {len(files)} files for ImmPort {accession}")
        else:
            logger.debug(f"ImmPort returned {response.status_code} for {accession}")
            
    except Exception as e:
        logger.warning(f"Error discovering ImmPort files for {accession}: {e}")
    
    return files


def discover_dataset_files(accession: str, source: str = None) -> List[DataFile]:
    """
    Discover available data files for a dataset.
    
    Routes to the appropriate repository-specific discovery function.
    
    Args:
        accession: Dataset accession ID
        source: Optional source hint (e.g., "geo", "arrayexpress", "pride")
    
    Returns:
        List of DataFile objects
    """
    accession_upper = accession.upper()
    
    # Use source hint if provided
    if source:
        source_lower = source.lower()
        if "flowrepo" in source_lower:
            return discover_flowrepository_files(accession)
        elif "immport" in source_lower:
            return discover_immport_files(accession)
    
    # Determine source from accession format
    if accession_upper.startswith("GSE") or accession_upper.startswith("GSM"):
        return discover_geo_files(accession)
    elif accession_upper.startswith("E-") or accession_upper.startswith("S-"):
        return discover_arrayexpress_files(accession)
    elif accession_upper.startswith("PXD"):
        return discover_pride_files(accession)
    elif accession_upper.startswith("FR-FCM"):
        return discover_flowrepository_files(accession)
    elif accession_upper.startswith("SDY"):
        return discover_immport_files(accession)
    else:
        # Try GEO as fallback for GSE-style accessions that might be formatted differently
        if "GSE" in accession_upper or accession_upper.startswith("G"):
            return discover_geo_files(accession)
        logger.debug(f"Unknown accession format: {accession}")
        return []


def filter_priority_files(files: List[DataFile], modality: str, max_files: int = 10) -> List[DataFile]:
    """
    Filter and prioritize files based on modality patterns.
    
    Args:
        files: List of discovered files
        modality: Dataset modality
        max_files: Maximum number of files to return
    
    Returns:
        Prioritized list of files matching the modality
    """
    # Score each file
    for f in files:
        matches, score = get_file_priority(f.filename, modality)
        f.modality_match = matches
        f.priority_score = score
    
    # Filter to only matching files
    matching = [f for f in files if f.modality_match]
    
    # Sort by priority (descending)
    matching.sort(key=lambda x: x.priority_score, reverse=True)
    
    # Return top files
    return matching[:max_files]


def download_file_to_gcs(url: str, dest_path: str, max_size_mb: int = 500) -> Optional[str]:
    """
    Download a file from URL and store in GCS.
    
    Args:
        url: Source file URL
        dest_path: Destination path in GCS (relative to bucket)
        max_size_mb: Maximum file size to download (in MB)
    
    Returns:
        GCS path if successful, None otherwise
    """
    client = get_storage_client()
    if not client:
        logger.warning(f"No storage client available for download")
        return None
    
    try:
        # Stream download to avoid memory issues
        response = requests.get(url, stream=True, timeout=300)
        response.raise_for_status()
        
        # Check content length
        content_length = int(response.headers.get("content-length", 0))
        max_bytes = max_size_mb * 1024 * 1024
        
        if content_length > max_bytes:
            logger.warning(f"File too large ({content_length / 1024 / 1024:.1f}MB > {max_size_mb}MB): {url}")
            return None
        
        # Upload to GCS in chunks
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(dest_path)
        
        # Use resumable upload for large files
        with blob.open("wb") as gcs_file:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    gcs_file.write(chunk)
                    downloaded += len(chunk)
                    
                    # Check size limit during download
                    if downloaded > max_bytes:
                        logger.warning(f"Download exceeded size limit: {url}")
                        return None
        
        gcs_path = f"gs://{BUCKET_NAME}/{dest_path}"
        logger.info(f"Downloaded {downloaded / 1024 / 1024:.1f}MB to {gcs_path}")
        return gcs_path
        
    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")
        return None


def download_dataset_files(
    accession: str, 
    modality: str, 
    source: str = None,
    max_files: int = 5,
    max_size_mb: int = 500
) -> List[DataFile]:
    """
    Discover and download priority data files for a dataset.
    
    Args:
        accession: Dataset accession ID
        modality: Dataset modality (for file prioritization)
        source: Optional source hint
        max_files: Maximum number of files to download
        max_size_mb: Maximum file size per file
    
    Returns:
        List of DataFile objects with download status
    """
    # Discover available files
    all_files = discover_dataset_files(accession, source)
    
    if not all_files:
        logger.info(f"No files found for {accession}")
        return []
    
    # Filter and prioritize
    priority_files = filter_priority_files(all_files, modality, max_files)
    
    if not priority_files:
        logger.info(f"No matching files for {accession} (modality: {modality})")
        return []
    
    logger.info(f"Downloading {len(priority_files)} priority files for {accession}")
    
    # Download each file
    for f in priority_files:
        # Build GCS destination path
        dest_path = f"data_files/{accession}/{f.filename}"
        
        gcs_path = download_file_to_gcs(f.url, dest_path, max_size_mb)
        
        if gcs_path:
            f.downloaded = True
            f.gcs_path = gcs_path
    
    downloaded_count = sum(1 for f in priority_files if f.downloaded)
    logger.info(f"Downloaded {downloaded_count}/{len(priority_files)} files for {accession}")
    
    # Save manifest and update metadata
    if downloaded_count > 0:
        save_file_manifest(accession, priority_files)
        update_dataset_with_files(accession, priority_files)
    
    return priority_files


def get_dataset_file_info(accession: str) -> Dict[str, Any]:
    """
    Get information about files already downloaded for a dataset.
    
    Args:
        accession: Dataset accession ID
    
    Returns:
        Dictionary with file information
    """
    client = get_storage_client()
    if not client:
        return {"files": [], "total_size": 0}
    
    try:
        bucket = client.bucket(BUCKET_NAME)
        prefix = f"data_files/{accession}/"
        
        blobs = list(bucket.list_blobs(prefix=prefix))
        
        files = []
        total_size = 0
        
        for blob in blobs:
            files.append({
                "filename": blob.name.split("/")[-1],
                "path": f"gs://{BUCKET_NAME}/{blob.name}",
                "size_bytes": blob.size,
                "updated": blob.updated.isoformat() if blob.updated else None,
            })
            total_size += blob.size or 0
        
        return {
            "files": files,
            "total_size": total_size,
            "file_count": len(files),
        }
        
    except Exception as e:
        logger.error(f"Error getting file info for {accession}: {e}")
        return {"files": [], "total_size": 0, "error": str(e)}


def update_dataset_with_files(accession: str, files: List[DataFile]) -> bool:
    """
    Update a dataset's metadata to include downloaded file information.
    
    Args:
        accession: Dataset accession ID
        files: List of DataFile objects that were downloaded
    
    Returns:
        True if successful
    """
    client = get_storage_client()
    if not client:
        return False
    
    try:
        bucket = client.bucket(BUCKET_NAME)
        
        # Find the dataset metadata blob
        prefix = f"harvest_results/"
        blobs = list(bucket.list_blobs(prefix=prefix))
        
        metadata_blob = None
        for blob in blobs:
            if blob.name.endswith(f"{accession}.json"):
                metadata_blob = blob
                break
        
        if not metadata_blob:
            logger.warning(f"No metadata found for {accession}")
            return False
        
        # Load existing metadata
        metadata = json.loads(metadata_blob.download_as_string())
        
        # Add file information
        metadata["data_files"] = {
            "download_time": datetime.utcnow().isoformat(),
            "files": [
                {
                    "filename": f.filename,
                    "gcs_path": f.gcs_path,
                    "file_type": f.file_type,
                    "priority_score": f.priority_score,
                    "size_bytes": f.size_bytes,
                }
                for f in files if f.downloaded and f.gcs_path
            ],
        }
        
        # Save updated metadata
        metadata_blob.upload_from_string(
            json.dumps(metadata, indent=2),
            content_type="application/json"
        )
        
        logger.info(f"Updated metadata for {accession} with {len(metadata['data_files']['files'])} files")
        return True
        
    except Exception as e:
        logger.error(f"Error updating metadata for {accession}: {e}")
        return False


def save_file_manifest(accession: str, files: List[DataFile]) -> bool:
    """
    Save a manifest of downloaded files for a dataset.
    
    This creates a separate manifest file that pairs the dataset
    with its downloaded data files.
    
    Args:
        accession: Dataset accession ID
        files: List of DataFile objects
    
    Returns:
        True if successful
    """
    client = get_storage_client()
    if not client:
        return False
    
    try:
        bucket = client.bucket(BUCKET_NAME)
        manifest_path = f"data_files/{accession}/manifest.json"
        blob = bucket.blob(manifest_path)
        
        manifest = {
            "accession": accession,
            "created": datetime.utcnow().isoformat(),
            "files": [
                {
                    "filename": f.filename,
                    "gcs_path": f.gcs_path,
                    "source_url": f.url,
                    "file_type": f.file_type,
                    "priority_score": f.priority_score,
                    "size_bytes": f.size_bytes,
                    "modality_match": f.modality_match,
                }
                for f in files
            ],
            "downloaded": [
                {
                    "filename": f.filename,
                    "gcs_path": f.gcs_path,
                }
                for f in files if f.downloaded and f.gcs_path
            ],
            "summary": {
                "total_found": len(files),
                "downloaded": sum(1 for f in files if f.downloaded),
                "failed": sum(1 for f in files if not f.downloaded),
            },
        }
        
        blob.upload_from_string(
            json.dumps(manifest, indent=2),
            content_type="application/json"
        )
        
        logger.info(f"Saved manifest for {accession}: {manifest['summary']}")
        return True
        
    except Exception as e:
        logger.error(f"Error saving manifest for {accession}: {e}")
        return False


# --- GEMINI AI CURATION ---

def get_gemini_client():
    """
    Get Gemini client for AI curation using Vertex AI.
    
    Requires GCP_PROJECT or GOOGLE_CLOUD_PROJECT environment variable.
    Uses Vertex AI backend with explicit project and location configuration.
    """
    # Robustly retrieve project ID
    project_id = os.environ.get("GCP_PROJECT") or os.environ.get("GOOGLE_CLOUD_PROJECT")
    
    if not project_id:
        logger.error(
            "Missing GCP project ID! Set either GCP_PROJECT or GOOGLE_CLOUD_PROJECT "
            "environment variable to enable Gemini AI curation."
        )
        return None
    
    try:
        # Initialize with explicit Vertex AI configuration
        client = genai.Client(
            vertexai=True,
            project=project_id,
            location=VERTEX_LOCATION,
        )
        logger.info(f"Gemini client initialized with project={project_id}, location={VERTEX_LOCATION}")
        return client
    except Exception as e:
        logger.error(f"Could not create Gemini client: {e}")
        return None


def _parse_first_json_object(text: str) -> Optional[Dict[str, Any]]:
    """
    Best-effort JSON parsing for model responses.
    
    - Handles leading/trailing whitespace and accidental markdown fences
    - Handles extra text after a valid JSON object via JSONDecoder.raw_decode
    """
    if not text:
        return None

    s = text.strip()

    # Defensive: strip accidental markdown fences if the model ignores instructions
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
        s = re.sub(r"\s*```$", "", s)
        s = s.strip()

    # Fast path
    try:
        parsed = json.loads(s)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass

    # Fallback: attempt to decode the first JSON object embedded in the string
    start = s.find("{")
    if start == -1:
        return None

    try:
        obj, _idx = json.JSONDecoder().raw_decode(s[start:])
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


@retry_with_backoff(max_retries=3, base_delay=5.0, retry_on_429=True)
def curate_with_gemini(result: HarvestResult) -> HarvestResult:
    """Use Gemini to classify and curate the dataset."""
    client = get_gemini_client()
    if not client:
        return result
    
    # Rate limit to avoid 429 RESOURCE_EXHAUSTED errors.
    # Configurable via CURATION_DELAY_SECONDS env var.
    # NOTE: keep this after client init so local/no-GCP runs don't pay the sleep cost.
    time.sleep(CURATION_DELAY_SECONDS)

    # Aggressively truncate input to minimize tokens
    title_short = (result.title[:80] if result.title else '').replace('\n', ' ').strip()
    summary_short = (result.summary[:150] if result.summary else '').replace('\n', ' ').strip()
    
    # Keep the model output *tiny* to avoid MAX_TOKENS truncation and JSON parse failures.
    # We intentionally omit free-text "reason" because it's the most common source of long outputs.
    prompt = f"""Return ONLY a single JSON object. No markdown. No extra text.

Schema (exact keys only):
{{"modality":"bulk_rna","functions":["bulk_rna"],"relevance":0}}

Rules:
- modality: one of bulk_rna,scrna,spatial,chipseq,atac,methylation,proteomics,flow,cytof,citeseq,multiome,repertoire,microbiome
- functions: array of 1-3 items chosen from the same list
- relevance: integer 0-10 for kidney transplant OR trained immunity relevance

Hint_modality:{result.modality}
Title:{title_short}
Desc:{summary_short}
"""
    
    try:
        # Disable safety filters completely for regulated research environment
        safety_settings = [
            types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
            types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),
            types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"),
            types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
        ]
        
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.0,  # Deterministic for classification
                max_output_tokens=512,  # Keep responses small; schema is tiny
                response_mime_type="application/json",  # Force native JSON mode
                safety_settings=safety_settings,
            )
        )
        
        # Check if response was blocked or truncated
        if response.text is None:
            # Log detailed info about why the response failed
            if response.candidates and len(response.candidates) > 0:
                candidate = response.candidates[0]
                finish_reason = getattr(candidate, 'finish_reason', 'UNKNOWN')
                safety_ratings = getattr(candidate, 'safety_ratings', [])
                
                # Determine error type
                finish_reason_str = str(finish_reason)
                if "MAX_TOKENS" in finish_reason_str:
                    error_msg = "Response truncated due to MAX_TOKENS limit"
                elif "SAFETY" in finish_reason_str:
                    error_msg = "Response blocked by safety filters"
                else:
                    error_msg = f"Response failed: {finish_reason}"
                
                logger.warning(
                    f"Gemini response issue for {result.accession}: "
                    f"finish_reason={finish_reason}, safety_ratings={safety_ratings}"
                )
                result.metadata["curation_error"] = error_msg
            else:
                logger.warning(f"Gemini response failed for {result.accession}: No candidates returned")
                result.metadata["curation_error"] = "No response candidates"
            
            return result
        
        # Parse JSON response (native JSON mode should return clean JSON)
        response_text = response.text.strip()
        
        curation = _parse_first_json_object(response_text)
        if curation is None:
            # Get finish reason if available for better diagnostics
            finish_reason = "UNKNOWN"
            if response.candidates and len(response.candidates) > 0:
                finish_reason = getattr(response.candidates[0], 'finish_reason', 'UNKNOWN')
            
            logger.warning(
                f"⚠️ JSON truncated/invalid for {result.accession}. Skipping curation. "
                f"Finish reason: {finish_reason}. "
                f"Raw_len={len(response_text)} "
                f"Head200={response_text[:200]!r} Tail200={response_text[-200:]!r}"
            )
            result.metadata["curation_error"] = f"JSON truncated (finish_reason={finish_reason})"
            return result  # Return uncurated result instead of crashing
        
        # Handle both old and new response formats
        graph_functions = curation.get("graph_functions", curation.get("functions", []))
        # Prepend "build_" and append "_graph" if missing
        normalized_functions = []
        for func in graph_functions[:3]:  # Limit to 3
            func_str = str(func).lower().strip()
            if not func_str.startswith("build_"):
                func_str = f"build_{func_str}"
            if not func_str.endswith("_graph"):
                func_str = f"{func_str}_graph"
            normalized_functions.append(func_str)
        
        result.graph_functions = normalized_functions
        result.curated = True
        
        relevance = curation.get("relevance_score", curation.get("relevance", 0))
        result.metadata.update({
            "gemini_curation": curation,
            "relevance_score": relevance,
            "is_multiomics": curation.get("is_multiomics", False),
            "confirmed_modality": curation.get("confirmed_modality", curation.get("modality", "")),
        })
        
        logger.info(f"Curated {result.accession}: relevance={relevance}, "
                   f"functions={result.graph_functions}")
        
    except Exception as e:
        error_str = str(e)
        logger.warning(f"Gemini curation failed for {result.accession}: {e}")
        
        # Detect quota exhaustion specifically and log actionable guidance
        if "RESOURCE_EXHAUSTED" in error_str or "429" in error_str:
            logger.error(
                f"Gemini quota hit for {result.accession}: {e}. "
                f"Using stable model (gemini-2.5-flash-lite) recommended."
            )
        
        result.metadata["curation_error"] = error_str
    
    return result


# --- CLAUDE AI CURATION VIA VERTEX AI (RECOMMENDED) ---

_anthropic_client = None

def get_anthropic_client():
    """
    Get or create Anthropic client for Claude curation via Vertex AI.
    
    Uses GCP service account credentials automatically - no separate API key needed.
    Claude models available in Vertex AI Model Garden: us-east5, europe-west1
    """
    global _anthropic_client
    
    if _anthropic_client is not None:
        return _anthropic_client
    
    project_id = os.environ.get("GCP_PROJECT") or os.environ.get("GOOGLE_CLOUD_PROJECT")
    if not project_id:
        logger.error(
            "Missing GCP_PROJECT! Set this environment variable to use Claude via Vertex AI. "
        )
        return None
    
    try:
        _anthropic_client = AnthropicVertex(
            project_id=project_id,
            region=CLAUDE_REGION,
        )
        logger.info(f"Anthropic Vertex client initialized: project={project_id}, region={CLAUDE_REGION}, model={CLAUDE_MODEL}")
        return _anthropic_client
    except Exception as e:
        logger.error(f"Could not create Anthropic Vertex client: {e}")
        return None


@retry_with_backoff(max_retries=3, base_delay=2.0, retry_on_429=True)
def curate_with_claude(result: HarvestResult) -> HarvestResult:
    """
    Use Claude to classify and curate the dataset.
    
    Claude 3.5 Haiku is optimized for:
    - Fast, reliable JSON output
    - Low latency (~200-500ms)
    - Low cost ($0.25/1M input, $1.25/1M output)
    - Excellent instruction following
    """
    client = get_anthropic_client()
    if not client:
        return result
    
    # Rate limit (Claude can handle much higher throughput than Gemini)
    time.sleep(CURATION_DELAY_SECONDS)
    
    # Prepare input
    title_short = (result.title[:100] if result.title else '').replace('\n', ' ').strip()
    summary_short = (result.summary[:200] if result.summary else '').replace('\n', ' ').strip()
    
    # Concise prompt optimized for Claude's instruction-following
    prompt = f"""Classify this dataset. Return ONLY valid JSON, nothing else.

Schema: {{"modality":"<type>","functions":["<type>",...],"relevance":<0-10>}}

Valid types: bulk_rna, scrna, spatial, chipseq, atac, methylation, proteomics, flow, cytof, citeseq, multiome, repertoire, microbiome

Context: Rate relevance 0-10 for kidney transplant OR trained immunity research.

Input modality hint: {result.modality}
Title: {title_short}
Description: {summary_short}"""

    try:
        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=128,  # Tiny output - just the JSON object
            temperature=0.0,  # Deterministic
            messages=[{"role": "user", "content": prompt}]
        )
        
        response_text = response.content[0].text.strip()
        
        # Parse JSON
        curation = _parse_first_json_object(response_text)
        if curation is None:
            stop_reason = response.stop_reason
            logger.warning(
                f"⚠️ Claude JSON invalid for {result.accession}. "
                f"stop_reason={stop_reason}, response={response_text[:200]!r}"
            )
            result.metadata["curation_error"] = f"JSON parse failed (stop={stop_reason})"
            return result
        
        # Normalize functions
        graph_functions = curation.get("graph_functions", curation.get("functions", []))
        normalized_functions = []
        for func in graph_functions[:3]:
            func_str = str(func).lower().strip()
            if not func_str.startswith("build_"):
                func_str = f"build_{func_str}"
            if not func_str.endswith("_graph"):
                func_str = f"{func_str}_graph"
            normalized_functions.append(func_str)
        
        result.graph_functions = normalized_functions
        result.curated = True
        
        relevance = curation.get("relevance_score", curation.get("relevance", 0))
        result.metadata.update({
            "claude_curation": curation,
            "curation_provider": "claude",
            "relevance_score": relevance,
            "confirmed_modality": curation.get("modality", ""),
        })
        
        logger.info(f"Curated {result.accession}: relevance={relevance}, functions={result.graph_functions}")
        
    except AnthropicRateLimitError as e:
        logger.warning(f"Claude rate limit for {result.accession}: {e}. Will retry with backoff.")
        raise  # Let retry_with_backoff handle it
        
    except AnthropicAPIError as e:
        logger.warning(f"Claude API error for {result.accession}: {e}")
        result.metadata["curation_error"] = str(e)
        
    except Exception as e:
        logger.warning(f"Claude curation failed for {result.accession}: {e}")
        result.metadata["curation_error"] = str(e)
    
    return result


def curate_dataset(result: HarvestResult) -> HarvestResult:
    """
    Route to the configured curation provider.
    
    Set CURATION_PROVIDER env var to 'claude' (recommended) or 'gemini'.
    """
    if CURATION_PROVIDER == "claude":
        return curate_with_claude(result)
    elif CURATION_PROVIDER == "gemini":
        return curate_with_gemini(result)
    else:
        logger.warning(f"Unknown CURATION_PROVIDER '{CURATION_PROVIDER}', using claude")
        return curate_with_claude(result)


# --- NCBI/GEO HARVESTER ---

@retry_with_backoff(max_retries=3, base_delay=2.0)
def ncbi_esearch(db: str, term: str, retmax: int = 100, retstart: int = 0) -> Dict[str, Any]:
    """Search NCBI database with global rate limiting."""
    _NCBI_RATE_LIMITER.acquire()  # Global thread-safe rate limiting
    handle = Entrez.esearch(
        db=db,
        term=term,
        retmax=retmax,
        retstart=retstart,
        usehistory="y",
        sort="relevance"
    )
    result = Entrez.read(handle)
    handle.close()
    return result


@retry_with_backoff(max_retries=3, base_delay=2.0)
def ncbi_esummary(db: str, id_list: List[str]) -> List[Dict[str, Any]]:
    """Get summaries for NCBI IDs with global rate limiting."""
    if not id_list:
        return []
    _NCBI_RATE_LIMITER.acquire()  # Global thread-safe rate limiting
    handle = Entrez.esummary(db=db, id=",".join(id_list))
    result = Entrez.read(handle)
    handle.close()
    return result if isinstance(result, list) else [result]


@retry_with_backoff(max_retries=3, base_delay=2.0)
def geo_fetch_details(gse_id: str) -> Optional[Dict[str, Any]]:
    """Fetch detailed GEO series information with global rate limiting."""
    try:
        _NCBI_RATE_LIMITER.acquire()
        handle = Entrez.esearch(db="gds", term=f"{gse_id}[Accession]", retmax=1)
        search_result = Entrez.read(handle)
        handle.close()
        
        if not search_result.get("IdList"):
            return None
        
        _NCBI_RATE_LIMITER.acquire()
        gds_id = search_result["IdList"][0]
        handle = Entrez.esummary(db="gds", id=gds_id)
        summary = Entrez.read(handle)
        handle.close()
        
        return summary[0] if summary else None
    except Exception as e:
        logger.warning(f"Could not fetch GEO details for {gse_id}: {e}")
        return None


def detect_modality_from_text(text: str) -> Tuple[Modality, float]:
    """Detect modality from description text with confidence score."""
    text_lower = text.lower()
    scores = {}
    
    for modality, keywords in MODALITY_QUERIES.items():
        score = sum(1 for kw in keywords if kw.lower() in text_lower)
        if score > 0:
            # Weight by specificity of matched keywords
            scores[modality] = score / len(keywords)
    
    if not scores:
        return Modality.BULK_RNA, 0.1  # Default fallback
    
    best_modality = max(scores, key=scores.get)
    return best_modality, scores[best_modality]


def harvest_ncbi_deep_paginated(db: str, query: str, modality_hint: Optional[Modality] = None,
                                max_results: int = 500, retstart_offset: int = 0,
                                seen_accessions: Optional[Set[str]] = None) -> Tuple[List[HarvestResult], int, bool]:
    """
    Deep harvest from NCBI databases with pagination support.
    
    Args:
        db: NCBI database (gds, sra, biosample, etc.)
        query: Search query
        modality_hint: Optional modality to guide classification
        max_results: Maximum results to fetch in this batch
        retstart_offset: Starting offset for pagination (for continuation)
        seen_accessions: Set of accessions to skip (already harvested)
    
    Returns:
        Tuple of (results list, next_offset, is_exhausted)
    """
    results = []
    retstart = retstart_offset
    batch_size = 50
    skipped_count = 0
    
    if seen_accessions is None:
        seen_accessions = set()
    
    logger.info(f"Harvesting NCBI {db} with query: {query[:100]}... (offset={retstart_offset})")
    
    try:
        # Initial search to get total count
        search_result = ncbi_esearch(db=db, term=query, retmax=1, retstart=0)
        total_count = int(search_result.get("Count", 0))
        logger.info(f"Found {total_count} records in NCBI {db}, starting at offset {retstart_offset}")
        
        # Check if we've already exhausted this query
        if retstart_offset >= total_count:
            logger.info(f"Already at end of results for {db} (offset {retstart_offset} >= total {total_count})")
            return results, retstart_offset, True
        
        # Iterate through results
        while retstart < min(total_count, retstart_offset + max_results):
            search_result = ncbi_esearch(
                db=db, 
                term=query, 
                retmax=batch_size, 
                retstart=retstart
            )
            
            id_list = search_result.get("IdList", [])
            if not id_list:
                break
            
            # Get summaries
            summaries = ncbi_esummary(db=db, id_list=id_list)
            
            for summary in summaries:
                try:
                    # Extract common fields
                    accession = summary.get("Accession", summary.get("Id", ""))
                    
                    # Skip if already seen
                    if accession in seen_accessions:
                        skipped_count += 1
                        continue
                    
                    title = summary.get("title", summary.get("Title", ""))
                    summary_text = summary.get("summary", summary.get("Summary", ""))
                    organism = summary.get("taxon", summary.get("Organism", "Homo sapiens"))
                    sample_count = int(summary.get("n_samples", summary.get("SampleCount", 1)))
                    
                    # Combine text for modality detection
                    combined_text = f"{title} {summary_text}"
                    
                    if modality_hint:
                        detected_modality = modality_hint
                        confidence = 0.8
                    else:
                        detected_modality, confidence = detect_modality_from_text(combined_text)
                    
                    # Sanitize raw_summary to avoid Biopython serialization issues
                    safe_raw_summary = clean_metadata(summary)
                    
                    result = HarvestResult(
                        source=f"ncbi_{db}",
                        modality=detected_modality.value,
                        accession=str(accession),
                        title=str(title)[:500],
                        summary=str(summary_text)[:5000],
                        organism=str(organism),
                        sample_count=sample_count,
                        metadata={
                            "ncbi_id": str(summary.get("Id", "")),
                            "modality_confidence": confidence,
                            "platform": str(summary.get("GPL", summary.get("Platform", ""))),
                            "pub_date": str(summary.get("PDAT", summary.get("PubDate", ""))),
                            "raw_summary": safe_raw_summary,
                        }
                    )
                    results.append(result)
                    
                except Exception as e:
                    logger.warning(f"Error processing NCBI record: {e}")
                    continue
            
            retstart += batch_size
            
            # Rate limiting safety
            if retstart % 200 == 0:
                logger.info(f"Processed {retstart}/{min(total_count, retstart_offset + max_results)} records, skipped {skipped_count} duplicates")
                time.sleep(1)
        
        # Determine if exhausted
        is_exhausted = retstart >= total_count
        
        if skipped_count > 0:
            logger.info(f"Skipped {skipped_count} already-seen accessions")
    
    except Exception as e:
        logger.error(f"NCBI harvest error: {e}")
        logger.debug(traceback.format_exc())
        return results, retstart, False
    
    return results, retstart, is_exhausted


def harvest_ncbi_deep(db: str, query: str, modality_hint: Optional[Modality] = None,
                      max_results: int = 500) -> List[HarvestResult]:
    """
    Deep harvest from NCBI databases (backward-compatible wrapper).
    
    Args:
        db: NCBI database (gds, sra, biosample, etc.)
        query: Search query
        modality_hint: Optional modality to guide classification
        max_results: Maximum results to fetch
    
    Returns:
        List of HarvestResult objects
    """
    results, _, _ = harvest_ncbi_deep_paginated(
        db=db, query=query, modality_hint=modality_hint, max_results=max_results
    )
    return results


def harvest_geo_modality(modality: Modality, disease_context: str = "trained immunity OR kidney transplant",
                         max_results: int = 200) -> List[HarvestResult]:
    """
    Harvest GEO datasets for a specific modality.
    
    Returns:
        List of HarvestResult objects
    """
    keywords = MODALITY_QUERIES.get(modality, [])
    if not keywords:
        return []
    
    # Build query with modality keywords and disease context
    keyword_query = " OR ".join(f'"{kw}"' for kw in keywords[:5])
    full_query = f"({keyword_query}) AND ({disease_context}) AND Homo sapiens[Organism]"
    
    return harvest_ncbi_deep(
        db="gds",
        query=full_query,
        modality_hint=modality,
        max_results=max_results,
        retstart_offset=retstart_offset,
        seen_accessions=seen_accessions
    )


# --- PRIDE PROTEOMICS HARVESTER ---

@rate_limit(0.5)
@retry_with_backoff(max_retries=3)
def fetch_pride_projects(query: str, page_size: int = 100, page: int = 0) -> Dict[str, Any]:
    """Fetch projects from PRIDE API."""
    url = "https://www.ebi.ac.uk/pride/ws/archive/v2/projects"
    params = {
        "keyword": query,
        "pageSize": page_size,
        "page": page,
    }
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def harvest_pride(keywords: List[str] = None, max_results: int = 200) -> List[HarvestResult]:
    """
    Harvest proteomics datasets from PRIDE database.
    
    Args:
        keywords: Search keywords (default: trained immunity + kidney transplant terms)
        max_results: Maximum results to fetch
    
    Returns:
        List of HarvestResult objects
    """
    if keywords is None:
        keywords = [
            "trained immunity", "kidney transplant", "renal transplant",
            "immune response", "monocyte", "macrophage", "cytokine"
        ]
    
    results = []
    seen_accessions = set()
    
    for keyword in keywords:
        try:
            logger.info(f"Searching PRIDE for: {keyword}")
            page = 0
            
            while len(results) < max_results:
                data = fetch_pride_projects(keyword, page_size=50, page=page)
                
                # Handle different PRIDE API response formats
                projects = []
                if isinstance(data, list):
                    # Direct list response (older API format)
                    projects = data
                elif isinstance(data, dict):
                    # Paginated response (newer API format)
                    projects = data.get("_embedded", {}).get("projects", [])
                    if not projects:
                        # Try alternative keys
                        projects = data.get("projects", data.get("compactProjects", []))
                
                if not projects:
                    break
                
                for project in projects:
                    # Handle both dict and other formats
                    if not isinstance(project, dict):
                        continue
                        
                    accession = project.get("accession", "")
                    if not accession or accession in seen_accessions:
                        continue
                    seen_accessions.add(accession)
                    
                    # Determine if phosphoproteomics
                    title_desc = f"{project.get('title', '')} {project.get('projectDescription', '')}"
                    if any(kw in title_desc.lower() for kw in ["phospho", "kinase", "phosphorylation"]):
                        modality = Modality.PHOSPHO
                    else:
                        modality = Modality.PROTEOMICS
                    
                    # Safely get organisms
                    organisms = project.get("organisms", ["Homo sapiens"])
                    if isinstance(organisms, list):
                        organism_str = ", ".join(str(o) for o in organisms)
                    else:
                        organism_str = str(organisms) if organisms else "Homo sapiens"
                    
                    result = HarvestResult(
                        source="pride",
                        modality=modality.value,
                        accession=accession,
                        title=str(project.get("title", ""))[:500],
                        summary=str(project.get("projectDescription", ""))[:5000],
                        organism=organism_str,
                        sample_count=len(project.get("sampleAttributes", [])) or 1,
                        metadata={
                            "submission_type": project.get("submissionType"),
                            "instruments": project.get("instruments", []),
                            "modifications": project.get("ptmNames", []),
                            "tissues": project.get("tissues", []),
                            "diseases": project.get("diseases", []),
                            "quantification": project.get("quantificationMethods", []),
                            "pub_date": project.get("publicationDate"),
                            "doi": project.get("doi"),
                        }
                    )
                    results.append(result)
                    
                    if len(results) >= max_results:
                        break
                
                page += 1
                
                # Safety check
                if page > 10:
                    break
        
        except Exception as e:
            logger.warning(f"PRIDE harvest error for '{keyword}': {e}")
            continue
    
    logger.info(f"Harvested {len(results)} proteomics datasets from PRIDE")
    return results


# --- METABOLOMICS WORKBENCH HARVESTER ---

@rate_limit(0.5)
@retry_with_backoff(max_retries=3)
def fetch_metabolomics_workbench(query: str) -> Dict[str, Any]:
    """Fetch from Metabolomics Workbench REST API."""
    url = f"https://www.metabolomicsworkbench.org/rest/study/study_id/{query}/summary"
    response = requests.get(url, timeout=30)
    if response.status_code == 200:
        return response.json()
    return {}


def harvest_metabolomics(disease_terms: List[str] = None, max_results: int = 100) -> List[HarvestResult]:
    """
    Harvest metabolomics datasets from Metabolomics Workbench.
    
    Note: Metabolomics Workbench has limited search API, so we use NCBI GEO 
    for metabolomics studies as a fallback.
    """
    results = []
    
    if disease_terms is None:
        disease_terms = ["immunity", "transplant", "kidney", "immune", "inflammation"]
    
    # Use NCBI/GEO for metabolomics studies
    for term in disease_terms[:3]:
        query = f'metabolomics AND "{term}" AND Homo sapiens[Organism]'
        geo_results = harvest_ncbi_deep(
            db="gds",
            query=query,
            modality_hint=Modality.METABOLOMICS,
            max_results=max_results // len(disease_terms)
        )
        results.extend(geo_results)
    
    # Deduplicate
    seen = set()
    unique_results = []
    for r in results:
        if r.accession not in seen:
            seen.add(r.accession)
            unique_results.append(r)
    
    logger.info(f"Harvested {len(unique_results)} metabolomics datasets")
    return unique_results


# --- IMMPORT HARVESTER (Immunophenotyping) ---
# Note: ImmPort API requires authentication. We use GEO as a fallback for immunology data.

def harvest_immport(assay_types: List[str] = None, max_results: int = 200) -> List[HarvestResult]:
    """
    Harvest immunology datasets using GEO as fallback.
    
    Note: ImmPort API requires authentication/registration. 
    This function uses GEO queries to find immunophenotyping data instead.
    
    Targets:
    - Flow cytometry (FACS)
    - CyTOF (mass cytometry)
    - ELISA/Luminex
    - Functional assays
    """
    logger.info("Using GEO fallback for immunophenotyping data (ImmPort requires authentication)")
    
    results = []
    
    # Define immunophenotyping queries for GEO
    immuno_queries = [
        ('"flow cytometry" AND (immunity OR transplant OR immune) AND Homo sapiens[Organism]', Modality.FLOW),
        ('"CyTOF" AND (immunity OR transplant OR immune) AND Homo sapiens[Organism]', Modality.CYTOF),
        ('"mass cytometry" AND (immunity OR transplant) AND Homo sapiens[Organism]', Modality.CYTOF),
        ('"ELISA" AND (cytokine OR immune) AND Homo sapiens[Organism]', Modality.CYTOKINE),
        ('"Luminex" AND (cytokine OR immune) AND Homo sapiens[Organism]', Modality.CYTOKINE),
        ('"ELISpot" AND (immune OR T cell) AND Homo sapiens[Organism]', Modality.FUNCTIONAL),
        ('"phospho-flow" AND immune AND Homo sapiens[Organism]', Modality.FUNCTIONAL),
    ]
    
    # Distribute max_results across queries
    per_query = max(10, max_results // len(immuno_queries))
    
    for query, modality in immuno_queries:
        try:
            geo_results = harvest_ncbi_deep(
                db="gds",
                query=query,
                modality_hint=modality,
                max_results=per_query
            )
            
            # Update source to indicate GEO fallback
            for r in geo_results:
                r.source = "immport_geo_fallback"
            
            results.extend(geo_results)
            
            if len(results) >= max_results:
                break
                
        except Exception as e:
            logger.warning(f"GEO immunophenotyping query failed: {e}")
            continue
    
    # Deduplicate
    seen = set()
    unique = []
    for r in results:
        if r.accession not in seen:
            seen.add(r.accession)
            unique.append(r)
    
    logger.info(f"Harvested {len(unique)} immunophenotyping datasets via GEO fallback")
    return unique[:max_results]


# --- FLOWREPOSITORY HARVESTER ---
# Note: FlowRepository SSL certificate is expired (as of Jan 2026).
# We now use GEO directly for flow cytometry data - more reliable and comprehensive.


def harvest_flowrepo(max_results: int = 200) -> List[HarvestResult]:
    """
    Harvest flow cytometry datasets via NCBI GEO.
    
    Note: FlowRepository's SSL certificate is expired, so we use GEO directly.
    GEO provides comprehensive flow cytometry datasets including:
    - Flow cytometry data (FACS)
    - Spectral flow cytometry
    - CyTOF/mass cytometry data
    """
    results = []
    
    logger.info("Harvesting flow cytometry data from GEO (FlowRepository SSL expired)")
    
    # GEO queries for flow cytometry data
    flow_queries = [
        ('"flow cytometry" AND (FACS OR sorting OR immune) AND Homo sapiens[Organism]', Modality.FLOW),
        ('"spectral flow" AND Homo sapiens[Organism]', Modality.FLOW),
        ('"CyTOF" AND Homo sapiens[Organism]', Modality.CYTOF),
        ('"mass cytometry" AND Homo sapiens[Organism]', Modality.CYTOF),
    ]
    
    # Distribute max_results across queries
    per_query = max(10, max_results // len(flow_queries))
    
    for query, modality in flow_queries:
        try:
            geo_results = harvest_ncbi_deep(
                db="gds",
                query=query,
                modality_hint=modality,
                max_results=per_query
            )
            for r in geo_results:
                r.source = "geo_flow"
            results.extend(geo_results)
            
            if len(results) >= max_results:
                break
                
        except Exception as e:
            logger.warning(f"GEO flow cytometry query failed: {e}")
            continue
    
    # Deduplicate
    seen = set()
    unique = []
    for r in results:
        if r.accession not in seen:
            seen.add(r.accession)
            unique.append(r)
    
    logger.info(f"Harvested {len(unique)} flow cytometry datasets from GEO")
    return unique[:max_results]


# --- ARRAYEXPRESS HARVESTER ---

@rate_limit(0.5)
@retry_with_backoff(max_retries=3)
def fetch_arrayexpress(query: str, page_size: int = 100) -> Dict[str, Any]:
    """Fetch from ArrayExpress/BioStudies API."""
    url = "https://www.ebi.ac.uk/biostudies/api/v1/search"
    params = {
        "query": query,
        "pageSize": page_size,
    }
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def harvest_arrayexpress(modalities: List[Modality] = None, max_results: int = 200) -> List[HarvestResult]:
    """
    Harvest from ArrayExpress/BioStudies for various modalities.
    
    Supports:
    - Microarray gene expression
    - RNA-seq
    - ChIP-seq
    - Methylation arrays
    - And more
    """
    results = []
    
    if modalities is None:
        modalities = [
            Modality.BULK_RNA,
            Modality.CHIPSEQ,
            Modality.METHYLATION,
            Modality.SCRNA,
        ]
    
    disease_terms = ["kidney transplant", "trained immunity", "immune"]
    
    for modality in modalities:
        keywords = MODALITY_QUERIES.get(modality, [])[:3]
        
        for term in disease_terms:
            query = f"{keywords[0]} AND {term}"
            
            try:
                data = fetch_arrayexpress(query, page_size=50)
                hits = data.get("hits", [])
                
                for hit in hits:
                    accession = hit.get("accession", "")
                    
                    result = HarvestResult(
                        source="arrayexpress",
                        modality=modality.value,
                        accession=accession,
                        title=hit.get("title", "")[:500],
                        summary=hit.get("description", "")[:5000],
                        organism=hit.get("organism", "Homo sapiens"),
                        sample_count=int(hit.get("samples", 1)),
                        metadata={
                            "technology": hit.get("technology"),
                            "experiment_type": hit.get("experimentType"),
                            "release_date": hit.get("releaseDate"),
                        }
                    )
                    results.append(result)
                    
                    if len(results) >= max_results:
                        break
                        
            except Exception as e:
                logger.warning(f"ArrayExpress harvest error for {modality}: {e}")
                continue
            
            if len(results) >= max_results:
                break
        
        if len(results) >= max_results:
            break
    
    # Deduplicate
    seen = set()
    unique = []
    for r in results:
        if r.accession not in seen:
            seen.add(r.accession)
            unique.append(r)
    
    logger.info(f"Harvested {len(unique)} datasets from ArrayExpress")
    return unique


# --- SPATIAL TRANSCRIPTOMICS (10X GENOMICS) HARVESTER ---

def harvest_spatial_transcriptomics(max_results: int = 100) -> List[HarvestResult]:
    """
    Harvest spatial transcriptomics datasets from GEO.
    
    Technologies covered:
    - 10x Visium
    - 10x Xenium
    - GeoMx
    - MERFISH
    - Slide-seq
    - CosMx
    """
    spatial_terms = [
        '"Visium"',
        '"spatial transcriptomics"',
        '"10x Xenium"',
        '"GeoMx"',
        '"MERFISH"',
        '"Slide-seq"',
        '"CosMx"',
    ]
    
    results = []
    
    for term in spatial_terms:
        query = f'{term} AND (immunity OR transplant OR immune OR kidney) AND Homo sapiens[Organism]'
        
        geo_results = harvest_ncbi_deep(
            db="gds",
            query=query,
            modality_hint=Modality.SPATIAL,
            max_results=max_results // len(spatial_terms)
        )
        results.extend(geo_results)
    
    # Deduplicate
    seen = set()
    unique = []
    for r in results:
        if r.accession not in seen:
            seen.add(r.accession)
            unique.append(r)
    
    logger.info(f"Harvested {len(unique)} spatial transcriptomics datasets")
    return unique


# --- TCR/BCR REPERTOIRE HARVESTER ---

def harvest_repertoire(max_results: int = 100) -> List[HarvestResult]:
    """
    Harvest TCR/BCR repertoire sequencing datasets.
    
    Sources:
    - GEO/SRA for TCR-seq, BCR-seq
    - Adaptive Biotechnologies immuneACCESS (requires registration)
    - VDJdb annotations
    """
    repertoire_terms = [
        '"TCR sequencing"',
        '"BCR sequencing"',
        '"VDJ repertoire"',
        '"immune repertoire"',
        '"T cell receptor sequencing"',
        '"B cell receptor sequencing"',
        '"10x VDJ"',
    ]
    
    results = []
    
    for term in repertoire_terms:
        query = f'{term} AND (immunity OR transplant OR immune) AND Homo sapiens[Organism]'
        
        geo_results = harvest_ncbi_deep(
            db="gds",
            query=query,
            modality_hint=Modality.REPERTOIRE,
            max_results=max_results // len(repertoire_terms)
        )
        results.extend(geo_results)
    
    # Deduplicate
    seen = set()
    unique = []
    for r in results:
        if r.accession not in seen:
            seen.add(r.accession)
            unique.append(r)
    
    logger.info(f"Harvested {len(unique)} repertoire datasets")
    return unique


# --- CITE-SEQ / MULTIOME HARVESTER ---

def harvest_citeseq_multiome(max_results: int = 100) -> List[HarvestResult]:
    """
    Harvest CITE-seq and Multiome datasets.
    
    Technologies:
    - CITE-seq (TotalSeq antibodies)
    - REAP-seq
    - DOGMA-seq
    - 10x Multiome (RNA + ATAC)
    - SHARE-seq
    """
    results = []
    
    # CITE-seq family
    citeseq_terms = ['"CITE-seq"', '"TotalSeq"', '"REAP-seq"', '"DOGMA-seq"']
    for term in citeseq_terms:
        query = f'{term} AND Homo sapiens[Organism]'
        geo_results = harvest_ncbi_deep(
            db="gds",
            query=query,
            modality_hint=Modality.CITESEQ,
            max_results=max_results // 4
        )
        results.extend(geo_results)
    
    # Multiome
    multiome_terms = ['"10x Multiome"', '"SHARE-seq"', '"scRNA ATAC"']
    for term in multiome_terms:
        query = f'{term} AND Homo sapiens[Organism]'
        geo_results = harvest_ncbi_deep(
            db="gds",
            query=query,
            modality_hint=Modality.MULTIOME,
            max_results=max_results // 3
        )
        results.extend(geo_results)
    
    # Deduplicate
    seen = set()
    unique = []
    for r in results:
        if r.accession not in seen:
            seen.add(r.accession)
            unique.append(r)
    
    logger.info(f"Harvested {len(unique)} CITE-seq/Multiome datasets")
    return unique


# --- PERTURB-SEQ / CRISPR SCREENS HARVESTER ---

def harvest_perturbseq(max_results: int = 100) -> List[HarvestResult]:
    """
    Harvest Perturb-seq and CRISPR screen datasets.
    
    Technologies:
    - Perturb-seq
    - CROP-seq
    - TAP-seq
    - Mosaic-seq
    - Pooled CRISPR screens
    """
    perturb_terms = [
        '"Perturb-seq"',
        '"CROP-seq"',
        '"TAP-seq"',
        '"CRISPR screen"',
        '"pooled CRISPR"',
        '"Mosaic-seq"',
    ]
    
    results = []
    
    for term in perturb_terms:
        query = f'{term} AND (immunity OR immune OR macrophage OR monocyte) AND Homo sapiens[Organism]'
        
        geo_results = harvest_ncbi_deep(
            db="gds",
            query=query,
            modality_hint=Modality.PERTURB,
            max_results=max_results // len(perturb_terms)
        )
        results.extend(geo_results)
    
    # Deduplicate
    seen = set()
    unique = []
    for r in results:
        if r.accession not in seen:
            seen.add(r.accession)
            unique.append(r)
    
    logger.info(f"Harvested {len(unique)} Perturb-seq/CRISPR screen datasets")
    return unique


# --- MICROBIOME HARVESTER ---

def harvest_microbiome(max_results: int = 100) -> List[HarvestResult]:
    """
    Harvest microbiome datasets (16S and shotgun metagenomics).
    
    Focuses on:
    - Gut microbiome in transplant
    - Microbiome-immunity interactions
    - Trained immunity and microbiome
    """
    results = []
    
    # 16S rRNA
    query_16s = '"16S rRNA" AND (transplant OR immunity OR immune) AND human[Organism]'
    results_16s = harvest_ncbi_deep(
        db="gds",
        query=query_16s,
        modality_hint=Modality.MICROBIOME_16S,
        max_results=max_results // 2
    )
    results.extend(results_16s)
    
    # Shotgun metagenomics
    query_meta = '"metagenomics" AND (transplant OR immunity OR immune OR kidney) AND human[Organism]'
    results_meta = harvest_ncbi_deep(
        db="gds",
        query=query_meta,
        modality_hint=Modality.METAGENOMICS,
        max_results=max_results // 2
    )
    results.extend(results_meta)
    
    # Deduplicate
    seen = set()
    unique = []
    for r in results:
        if r.accession not in seen:
            seen.add(r.accession)
            unique.append(r)
    
    logger.info(f"Harvested {len(unique)} microbiome datasets")
    return unique


# --- METHYLATION HARVESTER ---

def harvest_methylation(max_results: int = 100) -> List[HarvestResult]:
    """
    Harvest DNA methylation datasets.
    
    Technologies:
    - Illumina EPIC (850K) array
    - Illumina 450K array
    - WGBS (Whole Genome Bisulfite Sequencing)
    - RRBS (Reduced Representation Bisulfite Sequencing)
    """
    methylation_terms = [
        '"DNA methylation"',
        '"EPIC array"',
        '"450K methylation"',
        '"bisulfite sequencing"',
        '"WGBS"',
        '"RRBS"',
    ]
    
    results = []
    
    for term in methylation_terms:
        query = f'{term} AND (transplant OR immunity OR immune OR trained immunity) AND Homo sapiens[Organism]'
        
        geo_results = harvest_ncbi_deep(
            db="gds",
            query=query,
            modality_hint=Modality.METHYLATION,
            max_results=max_results // len(methylation_terms)
        )
        results.extend(geo_results)
    
    # Deduplicate
    seen = set()
    unique = []
    for r in results:
        if r.accession not in seen:
            seen.add(r.accession)
            unique.append(r)
    
    logger.info(f"Harvested {len(unique)} methylation datasets")
    return unique


# --- HI-C / 3D CHROMATIN HARVESTER ---

def harvest_hic(max_results: int = 50) -> List[HarvestResult]:
    """
    Harvest Hi-C and 3D chromatin datasets.
    
    Technologies:
    - Hi-C
    - Micro-C
    - HiChIP
    - PLAC-seq
    - Capture Hi-C
    """
    hic_terms = [
        '"Hi-C"',
        '"Micro-C"',
        '"HiChIP"',
        '"3D chromatin"',
        '"chromatin conformation"',
    ]
    
    results = []
    
    for term in hic_terms:
        query = f'{term} AND (immune OR macrophage OR monocyte OR T cell) AND Homo sapiens[Organism]'
        
        geo_results = harvest_ncbi_deep(
            db="gds",
            query=query,
            modality_hint=Modality.HIC,
            max_results=max_results // len(hic_terms)
        )
        results.extend(geo_results)
    
    # Deduplicate
    seen = set()
    unique = []
    for r in results:
        if r.accession not in seen:
            seen.add(r.accession)
            unique.append(r)
    
    logger.info(f"Harvested {len(unique)} Hi-C/3D chromatin datasets")
    return unique


# --- GLYCOMICS HARVESTER ---

def harvest_glycomics(max_results: int = 50) -> List[HarvestResult]:
    """
    Harvest glycomics datasets.
    
    Sources:
    - GEO/ArrayExpress for glycomics arrays
    - GlyGen database
    - UniCarb-DB
    """
    glycomics_terms = [
        '"glycomics"',
        '"glycan array"',
        '"glycoproteomics"',
        '"lectin array"',
        '"glycosylation"',
    ]
    
    results = []
    
    for term in glycomics_terms:
        query = f'{term} AND (immune OR antibody OR IgG) AND Homo sapiens[Organism]'
        
        geo_results = harvest_ncbi_deep(
            db="gds",
            query=query,
            modality_hint=Modality.GLYCOMICS,
            max_results=max_results // len(glycomics_terms)
        )
        results.extend(geo_results)
    
    # Deduplicate
    seen = set()
    unique = []
    for r in results:
        if r.accession not in seen:
            seen.add(r.accession)
            unique.append(r)
    
    logger.info(f"Harvested {len(unique)} glycomics datasets")
    return unique


# --- LIPIDOMICS HARVESTER ---

def harvest_lipidomics(max_results: int = 50) -> List[HarvestResult]:
    """Harvest lipidomics datasets."""
    results = []
    
    query = '"lipidomics" OR "lipid profiling" AND (immune OR inflammation OR macrophage) AND Homo sapiens[Organism]'
    
    geo_results = harvest_ncbi_deep(
        db="gds",
        query=query,
        modality_hint=Modality.LIPIDOMICS,
        max_results=max_results
    )
    results.extend(geo_results)
    
    # Deduplicate
    seen = set()
    unique = []
    for r in results:
        if r.accession not in seen:
            seen.add(r.accession)
            unique.append(r)
    
    logger.info(f"Harvested {len(unique)} lipidomics datasets")
    return unique


# --- CNV ARRAYS HARVESTER ---

def harvest_cnv(max_results: int = 50) -> List[HarvestResult]:
    """
    Harvest CNV (Copy Number Variation) array datasets.
    
    Sources:
    - GEO/ArrayExpress for CNV arrays (aCGH, SNP arrays with CNV)
    """
    cnv_terms = [
        '"copy number variation"',
        '"CNV array"',
        '"array CGH"',
        '"aCGH"',
        '"copy number analysis"',
    ]
    
    results = []
    
    for term in cnv_terms:
        query = f'{term} AND (kidney OR transplant OR immune) AND Homo sapiens[Organism]'
        
        geo_results = harvest_ncbi_deep(
            db="gds",
            query=query,
            modality_hint=Modality.CNV,
            max_results=max_results // len(cnv_terms)
        )
        results.extend(geo_results)
    
    # Deduplicate
    seen = set()
    unique = []
    for r in results:
        if r.accession not in seen:
            seen.add(r.accession)
            unique.append(r)
    
    logger.info(f"Harvested {len(unique)} CNV array datasets")
    return unique


# --- CYTOF / MASS CYTOMETRY HARVESTER ---

def harvest_cytof(max_results: int = 100) -> List[HarvestResult]:
    """
    Harvest CyTOF (mass cytometry) datasets.
    
    Sources:
    - GEO for CyTOF/mass cytometry data
    - FlowRepository (some CyTOF data)
    - Cytobank public datasets
    """
    cytof_terms = [
        '"mass cytometry"',
        '"CyTOF"',
        '"Helios"',
        '"metal-tagged antibodies"',
        '"cytometry by time of flight"',
    ]
    
    results = []
    
    for term in cytof_terms:
        query = f'{term} AND (immune OR kidney OR transplant OR monocyte OR macrophage) AND Homo sapiens[Organism]'
        
        geo_results = harvest_ncbi_deep(
            db="gds",
            query=query,
            modality_hint=Modality.CYTOF,
            max_results=max_results // len(cytof_terms)
        )
        results.extend(geo_results)
    
    # Also try general search with broader terms
    broad_query = '("CyTOF" OR "mass cytometry") AND (trained immunity OR kidney transplant) AND Homo sapiens[Organism]'
    broad_results = harvest_ncbi_deep(
        db="gds",
        query=broad_query,
        modality_hint=Modality.CYTOF,
        max_results=max_results // 2
    )
    results.extend(broad_results)
    
    # Deduplicate
    seen = set()
    unique = []
    for r in results:
        if r.accession not in seen:
            seen.add(r.accession)
            unique.append(r)
    
    logger.info(f"Harvested {len(unique)} CyTOF/mass cytometry datasets")
    return unique


# --- MULTIPLEX IHC/IF HARVESTER (CODEX, MIBI, IMC) ---

def harvest_multiplex_imaging(max_results: int = 100) -> List[HarvestResult]:
    """
    Harvest multiplex imaging datasets (CODEX, MIBI, IMC, Phenocycler).
    
    Sources:
    - GEO for multiplex imaging data
    - HuBMAP portal
    - Specialized imaging repositories
    """
    imaging_terms = [
        '"CODEX"',
        '"MIBI"',
        '"IMC"',
        '"imaging mass cytometry"',
        '"multiplex immunofluorescence"',
        '"multiplex IHC"',
        '"Phenocycler"',
        '"Akoya"',
        '"Vectra"',
    ]
    
    results = []
    
    for term in imaging_terms:
        query = f'{term} AND (kidney OR transplant OR immune OR tissue) AND Homo sapiens[Organism]'
        
        geo_results = harvest_ncbi_deep(
            db="gds",
            query=query,
            modality_hint=Modality.MIHC,
            max_results=max_results // len(imaging_terms)
        )
        results.extend(geo_results)
    
    # Deduplicate
    seen = set()
    unique = []
    for r in results:
        if r.accession not in seen:
            seen.add(r.accession)
            unique.append(r)
    
    logger.info(f"Harvested {len(unique)} multiplex imaging datasets")
    return unique


# --- FLUXOMICS HARVESTER ---

def harvest_fluxomics(max_results: int = 50) -> List[HarvestResult]:
    """
    Harvest fluxomics / metabolic flux datasets.
    
    Sources:
    - GEO for isotope tracing studies
    - Metabolomics Workbench
    """
    flux_terms = [
        '"metabolic flux"',
        '"13C tracing"',
        '"isotope tracing"',
        '"Seahorse"',
        '"extracellular acidification"',
        '"oxygen consumption rate"',
        '"fluxomics"',
    ]
    
    results = []
    
    for term in flux_terms:
        query = f'{term} AND (immune OR macrophage OR monocyte OR trained immunity) AND Homo sapiens[Organism]'
        
        geo_results = harvest_ncbi_deep(
            db="gds",
            query=query,
            modality_hint=Modality.FLUXOMICS,
            max_results=max_results // len(flux_terms)
        )
        results.extend(geo_results)
    
    # Deduplicate
    seen = set()
    unique = []
    for r in results:
        if r.accession not in seen:
            seen.add(r.accession)
            unique.append(r)
    
    logger.info(f"Harvested {len(unique)} fluxomics datasets")
    return unique


# --- eccDNA / circRNA HARVESTER ---

def harvest_eccdna(max_results: int = 50) -> List[HarvestResult]:
    """
    Harvest eccDNA (extrachromosomal circular DNA) and circRNA datasets.
    
    Sources:
    - GEO for eccDNA/circRNA studies
    - CircBase references
    """
    eccdna_terms = [
        '"eccDNA"',
        '"extrachromosomal circular DNA"',
        '"circRNA"',
        '"circular RNA"',
        '"ecDNA"',
    ]
    
    results = []
    
    for term in eccdna_terms:
        query = f'{term} AND (immune OR cancer OR kidney) AND Homo sapiens[Organism]'
        
        geo_results = harvest_ncbi_deep(
            db="gds",
            query=query,
            modality_hint=Modality.ECCDNA,
            max_results=max_results // len(eccdna_terms)
        )
        results.extend(geo_results)
    
    # Deduplicate
    seen = set()
    unique = []
    for r in results:
        if r.accession not in seen:
            seen.add(r.accession)
            unique.append(r)
    
    logger.info(f"Harvested {len(unique)} eccDNA/circRNA datasets")
    return unique


# --- FUNCTIONAL ASSAYS HARVESTER ---

def harvest_functional_assays(max_results: int = 50) -> List[HarvestResult]:
    """
    Harvest functional assay datasets (phospho-flow, cytokine secretion, etc.).
    
    Sources:
    - GEO/ImmPort for functional assay data
    """
    functional_terms = [
        '"phospho-flow"',
        '"ELISpot"',
        '"cytokine secretion"',
        '"killing assay"',
        '"degranulation assay"',
        '"functional assay" AND immune',
    ]
    
    results = []
    
    for term in functional_terms:
        query = f'{term} AND Homo sapiens[Organism]'
        
        geo_results = harvest_ncbi_deep(
            db="gds",
            query=query,
            modality_hint=Modality.FUNCTIONAL,
            max_results=max_results // len(functional_terms)
        )
        results.extend(geo_results)
    
    # Deduplicate
    seen = set()
    unique = []
    for r in results:
        if r.accession not in seen:
            seen.add(r.accession)
            unique.append(r)
    
    logger.info(f"Harvested {len(unique)} functional assay datasets")
    return unique


# --- KPMP (Kidney Precision Medicine Project) HARVESTER ---

@rate_limit(1.0)
def kpmp_api_search(endpoint: str, params: Dict = None) -> Optional[Dict]:
    """Query KPMP API."""
    base_url = "https://atlas.kpmp.org/api"
    try:
        response = requests.get(
            f"{base_url}/{endpoint}",
            params=params or {},
            timeout=30,
            headers={"Accept": "application/json"}
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.warning(f"KPMP API error: {e}")
        return None


def harvest_kpmp(max_results: int = 200) -> List[HarvestResult]:
    """
    Harvest datasets from Kidney Precision Medicine Project (KPMP).
    
    KPMP provides single-cell, spatial, and bulk transcriptomics data
    specifically for kidney disease research.
    
    Note: KPMP data is publicly available without access requests.
    """
    results = []
    
    # KPMP has several data types we want to capture
    kpmp_modalities = [
        ("single-cell RNA-seq", Modality.SCRNA, "build_scrna_graph"),
        ("single-nucleus RNA-seq", Modality.SCRNA, "build_scrna_graph"),
        ("spatial transcriptomics", Modality.SPATIAL, "build_spatial_graph"),
        ("CODEX", Modality.MIHC, "build_spatial_graph"),
        ("proteomics", Modality.PROTEOMICS, "build_proteomics_graph"),
        ("metabolomics", Modality.METABOLOMICS, "build_metabolomics_graph"),
    ]
    
    # Try KPMP atlas API
    try:
        # Search for available datasets
        # KPMP provides data through their atlas portal
        atlas_data = kpmp_api_search("datasets")
        
        if atlas_data and isinstance(atlas_data, list):
            for item in atlas_data[:max_results]:
                dataset_id = item.get("id", item.get("datasetId", ""))
                title = item.get("title", item.get("name", "KPMP Dataset"))
                description = item.get("description", "")
                data_type = item.get("dataType", item.get("assayType", ""))
                
                # Determine modality from data type
                modality = Modality.SCRNA  # Default
                graph_func = "build_scrna_graph"
                
                for mod_name, mod_enum, func in kpmp_modalities:
                    if mod_name.lower() in data_type.lower() or mod_name.lower() in title.lower():
                        modality = mod_enum
                        graph_func = func
                        break
                
                result = HarvestResult(
                    source="kpmp",
                    accession=f"KPMP-{dataset_id}" if dataset_id else f"KPMP-{hash(title) % 100000}",
                    title=title,
                    description=description,
                    modality=modality,
                    organism="Homo sapiens",
                    metadata={
                        "repository": "KPMP",
                        "data_type": data_type,
                        "url": f"https://atlas.kpmp.org/explorer/{dataset_id}" if dataset_id else "https://atlas.kpmp.org",
                        "access_type": "public",
                    },
                    graph_functions=[graph_func],
                    relevance_score=8,  # High relevance for kidney research
                )
                results.append(result)
                
    except Exception as e:
        logger.warning(f"KPMP API harvest failed: {e}")
    
    # Fallback: Search GEO for KPMP-related datasets
    if len(results) < max_results // 2:
        logger.info("Using GEO fallback for KPMP datasets...")
        kpmp_geo_query = '("KPMP" OR "Kidney Precision Medicine Project") AND Homo sapiens[Organism]'
        
        geo_results = harvest_ncbi_deep(
            db="gds",
            query=kpmp_geo_query,
            modality_hint=Modality.SCRNA,
            max_results=max_results - len(results)
        )
        
        # Mark these as KPMP-related
        for r in geo_results:
            r.metadata["kpmp_related"] = True
            r.relevance_score = max(r.relevance_score, 7)
        
        results.extend(geo_results)
    
    # Deduplicate
    seen = set()
    unique = []
    for r in results:
        if r.accession not in seen:
            seen.add(r.accession)
            unique.append(r)
    
    logger.info(f"Harvested {len(unique)} KPMP datasets")
    return unique


# --- HCA (Human Cell Atlas) KIDNEY HARVESTER ---

@rate_limit(1.0)
def hca_api_search(filters: Dict = None, max_results: int = 100) -> List[Dict]:
    """Query Human Cell Atlas Data Portal API."""
    base_url = "https://service.azul.data.humancellatlas.org/index/projects"
    
    try:
        # HCA uses a catalog-based API
        params = {
            "catalog": "dcp1",  # Data Coordination Platform
            "size": min(max_results, 100),
        }
        
        if filters:
            params["filters"] = json.dumps(filters)
        
        response = requests.get(
            base_url,
            params=params,
            timeout=30,
            headers={"Accept": "application/json"}
        )
        response.raise_for_status()
        data = response.json()
        
        return data.get("hits", [])
    except Exception as e:
        logger.warning(f"HCA API error: {e}")
        return []


def harvest_hca_kidney(max_results: int = 200) -> List[HarvestResult]:
    """
    Harvest kidney-related datasets from Human Cell Atlas.
    
    HCA provides high-quality single-cell reference data.
    Focuses on kidney tissue atlas and related immune studies.
    """
    results = []
    
    # HCA filters for kidney-related data
    kidney_filters = {
        "organ": {"is": ["kidney", "renal"]},
        "specimenOrgan": {"is": ["kidney"]},
    }
    
    try:
        # Search for kidney projects
        hca_data = hca_api_search(filters=kidney_filters, max_results=max_results)
        
        for project in hca_data:
            project_info = project.get("projects", [{}])[0]
            
            project_id = project_info.get("projectId", "")
            title = project_info.get("projectTitle", "HCA Kidney Project")
            description = project_info.get("projectDescription", "")
            
            # Get cell count and other metadata
            cell_count = project.get("cellSuspensions", {}).get("totalCells", 0)
            organs = project.get("samples", {}).get("organ", [])
            
            # Determine modality
            library_construction = project.get("protocols", {}).get("libraryConstructionApproach", [])
            
            modality = Modality.SCRNA
            graph_func = "build_scrna_graph"
            
            if any("10x" in lc.lower() or "smart-seq" in lc.lower() for lc in library_construction):
                modality = Modality.SCRNA
                graph_func = "build_scrna_graph"
            elif any("spatial" in lc.lower() or "visium" in lc.lower() for lc in library_construction):
                modality = Modality.SPATIAL
                graph_func = "build_spatial_graph"
            
            result = HarvestResult(
                source="hca",
                accession=f"HCA-{project_id}" if project_id else f"HCA-{hash(title) % 100000}",
                title=title,
                description=description[:1000] if description else "",
                modality=modality,
                organism="Homo sapiens",
                metadata={
                    "repository": "Human Cell Atlas",
                    "cell_count": cell_count,
                    "organs": organs,
                    "library_methods": library_construction,
                    "url": f"https://data.humancellatlas.org/explore/projects/{project_id}" if project_id else "https://data.humancellatlas.org",
                    "access_type": "public",
                },
                graph_functions=[graph_func],
                relevance_score=8,  # High relevance for kidney research
            )
            results.append(result)
            
    except Exception as e:
        logger.warning(f"HCA API harvest failed: {e}")
    
    # Also search for immune-related HCA projects
    immune_filters = {
        "organ": {"is": ["blood", "bone marrow", "lymph node", "spleen"]},
    }
    
    try:
        immune_data = hca_api_search(filters=immune_filters, max_results=max_results // 2)
        
        for project in immune_data:
            project_info = project.get("projects", [{}])[0]
            
            project_id = project_info.get("projectId", "")
            title = project_info.get("projectTitle", "HCA Immune Project")
            description = project_info.get("projectDescription", "")
            
            # Check if relevant to trained immunity
            relevance = 5
            if any(term in (title + description).lower() for term in ["monocyte", "macrophage", "innate", "trained"]):
                relevance = 8
            
            result = HarvestResult(
                source="hca",
                accession=f"HCA-{project_id}" if project_id else f"HCA-{hash(title) % 100000}",
                title=title,
                description=description[:1000] if description else "",
                modality=Modality.SCRNA,
                organism="Homo sapiens",
                metadata={
                    "repository": "Human Cell Atlas",
                    "focus": "immune",
                    "url": f"https://data.humancellatlas.org/explore/projects/{project_id}" if project_id else "https://data.humancellatlas.org",
                    "access_type": "public",
                },
                graph_functions=["build_scrna_graph"],
                relevance_score=relevance,
            )
            results.append(result)
            
    except Exception as e:
        logger.warning(f"HCA immune harvest failed: {e}")
    
    # Fallback: Search GEO for HCA-related datasets
    if len(results) < max_results // 4:
        logger.info("Using GEO fallback for HCA-related datasets...")
        hca_geo_query = '("Human Cell Atlas" OR "HCA") AND kidney AND Homo sapiens[Organism]'
        
        geo_results = harvest_ncbi_deep(
            db="gds",
            query=hca_geo_query,
            modality_hint=Modality.SCRNA,
            max_results=max_results // 2
        )
        
        for r in geo_results:
            r.metadata["hca_related"] = True
        
        results.extend(geo_results)
    
    # Deduplicate
    seen = set()
    unique = []
    for r in results:
        if r.accession not in seen:
            seen.add(r.accession)
            unique.append(r)
    
    logger.info(f"Harvested {len(unique)} HCA kidney/immune datasets")
    return unique


# --- MASTER HARVESTER ORCHESTRATOR ---

def harvest_all_modalities(max_per_modality: int = 100, 
                           enable_curation: bool = True,
                           modalities: List[str] = None,
                           max_curations: int = 0) -> Dict[str, List[HarvestResult]]:
    """
    Orchestrate harvesting across all modalities.
    
    IMPORTANT: NCBI-dependent harvesters are run SEQUENTIALLY to respect rate limits.
    Independent harvesters (PRIDE, ArrayExpress) run in parallel.
    
    Args:
        max_per_modality: Maximum results per modality
        enable_curation: Whether to run Gemini AI curation
        modalities: Optional list of specific modalities to harvest
        max_curations: Maximum number of items to curate (0 = use MAX_CURATIONS_PER_RUN default)
    
    Returns:
        Dictionary mapping modality names to lists of HarvestResult
    """
    all_results: Dict[str, List[HarvestResult]] = {}
    start_time = time.time()
    
    # NCBI-dependent harvesters - MUST run sequentially to avoid 429 errors
    ncbi_harvesters = [
        # Transcriptomics
        ("geo_bulk_rna", lambda: harvest_geo_modality(Modality.BULK_RNA, max_results=max_per_modality)),
        ("geo_scrna", lambda: harvest_geo_modality(Modality.SCRNA, max_results=max_per_modality)),
        ("spatial", lambda: harvest_spatial_transcriptomics(max_results=max_per_modality)),
        ("geo_mirna", lambda: harvest_geo_modality(Modality.MIRNA, max_results=max_per_modality // 2)),
        ("geo_longread", lambda: harvest_geo_modality(Modality.LONGREAD, max_results=max_per_modality // 2)),
        ("eccdna", lambda: harvest_eccdna(max_results=max_per_modality // 2)),
        
        # Genomics
        ("geo_wgs", lambda: harvest_geo_modality(Modality.WGS, max_results=max_per_modality // 2)),
        ("geo_wes", lambda: harvest_geo_modality(Modality.WES, max_results=max_per_modality // 2)),
        ("geo_hla", lambda: harvest_geo_modality(Modality.HLA, max_results=max_per_modality // 2)),
        ("geo_snp", lambda: harvest_geo_modality(Modality.SNP, max_results=max_per_modality // 2)),
        ("cnv", lambda: harvest_cnv(max_results=max_per_modality // 2)),
        
        # Epigenomics
        ("geo_chipseq", lambda: harvest_geo_modality(Modality.CHIPSEQ, max_results=max_per_modality)),
        ("geo_atac", lambda: harvest_geo_modality(Modality.ATAC, max_results=max_per_modality)),
        ("methylation", lambda: harvest_methylation(max_results=max_per_modality)),
        ("hic", lambda: harvest_hic(max_results=max_per_modality // 2)),
        
        # Proteomics & Signaling
        ("geo_cytokine", lambda: harvest_geo_modality(Modality.CYTOKINE, max_results=max_per_modality // 2)),
        ("glycomics", lambda: harvest_glycomics(max_results=max_per_modality // 2)),
        
        # Metabolism
        ("metabolomics", lambda: harvest_metabolomics(max_results=max_per_modality)),
        ("lipidomics", lambda: harvest_lipidomics(max_results=max_per_modality // 2)),
        ("fluxomics", lambda: harvest_fluxomics(max_results=max_per_modality // 2)),
        
        # Immunophenotyping
        ("immport", lambda: harvest_immport(max_results=max_per_modality)),
        ("flowrepo", lambda: harvest_flowrepo(max_results=max_per_modality)),
        ("cytof", lambda: harvest_cytof(max_results=max_per_modality)),
        ("multiplex_imaging", lambda: harvest_multiplex_imaging(max_results=max_per_modality)),
        ("functional_assays", lambda: harvest_functional_assays(max_results=max_per_modality // 2)),
        
        # Repertoire & Microbiome
        ("repertoire", lambda: harvest_repertoire(max_results=max_per_modality)),
        ("microbiome", lambda: harvest_microbiome(max_results=max_per_modality)),
        
        # Multi-omics
        ("citeseq_multiome", lambda: harvest_citeseq_multiome(max_results=max_per_modality)),
        ("perturbseq", lambda: harvest_perturbseq(max_results=max_per_modality // 2)),
    ]
    
    # Independent harvesters - can run in parallel (separate APIs with own rate limits)
    independent_harvesters = [
        ("pride", lambda: harvest_pride(max_results=max_per_modality)),
        ("arrayexpress", lambda: harvest_arrayexpress(max_results=max_per_modality)),
        ("kpmp", lambda: harvest_kpmp(max_results=max_per_modality)),
        ("hca", lambda: harvest_hca_kidney(max_results=max_per_modality)),
    ]
    
    # Filter harvesters if specific modalities requested
    if modalities:
        modalities_set = set(m.lower() for m in modalities)
        ncbi_harvesters = [(name, func) for name, func in ncbi_harvesters if name.lower() in modalities_set]
        independent_harvesters = [(name, func) for name, func in independent_harvesters if name.lower() in modalities_set]
    
    # Helper to run a single harvester with error handling
    def run_harvester(name: str, harvester_func: Callable) -> Tuple[str, List[HarvestResult]]:
        try:
            logger.info(f"Starting harvester: {name}")
            results = harvester_func()
            logger.info(f"Harvester {name} completed with {len(results)} results")
            return name, results
        except Exception as e:
            logger.error(f"Harvester {name} failed: {e}")
            logger.debug(traceback.format_exc())
            return name, []
    
    # Run NCBI harvesters SEQUENTIALLY to respect global rate limit
    logger.info(f"Running {len(ncbi_harvesters)} NCBI-dependent harvesters sequentially...")
    for name, func in ncbi_harvesters:
        # Check runtime limit
        if time.time() - start_time > MAX_RUNTIME_SECONDS - 300:  # 5 min buffer
            logger.warning("Approaching runtime limit, stopping NCBI harvesters")
            break
        
        harvester_name, results = run_harvester(name, func)
        all_results[harvester_name] = results
        
        # Brief pause between harvesters to let rate limit settle
        time.sleep(0.5)
    
    # Run independent harvesters in parallel (they have separate APIs)
    if independent_harvesters and (time.time() - start_time < MAX_RUNTIME_SECONDS - 300):
        logger.info(f"Running {len(independent_harvesters)} independent harvesters in parallel...")
        
        with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(independent_harvesters))) as executor:
            futures = {
                executor.submit(run_harvester, name, func): name 
                for name, func in independent_harvesters
            }
            
            for future in as_completed(futures):
                name, results = future.result()
                all_results[name] = results
    
    # Apply AI curation if enabled (parallel processing)
    if enable_curation:
        # Determine curation limit: use parameter if > 0, else use env default
        curation_limit = max_curations if max_curations > 0 else MAX_CURATIONS_PER_RUN
        
        # Collect all uncurated results with their source info
        uncurated_items: List[Tuple[str, int, HarvestResult]] = []
        for source_name, results in all_results.items():
            for i, result in enumerate(results):
                if not result.curated:
                    uncurated_items.append((source_name, i, result))
        
        total_uncurated = len(uncurated_items)
        effective_limit = min(curation_limit, total_uncurated) if curation_limit > 0 else total_uncurated
        items_to_curate = uncurated_items[:effective_limit]
        
        logger.info(
            f"Starting AI curation [{CURATION_PROVIDER}]: {total_uncurated} uncurated, "
            f"processing {effective_limit} with {CURATION_WORKERS} parallel workers"
        )
        
        curated_count = 0
        failed_count = 0
        curation_start = time.time()
        
        def curate_single(item: Tuple[str, int, HarvestResult]) -> Tuple[str, int, HarvestResult, bool]:
            """Curate a single item, return (source, index, result, success)."""
            source_name, idx, result = item
            try:
                curated = curate_dataset(result)
                return (source_name, idx, curated, True)
            except Exception as e:
                logger.warning(f"Curation failed for {result.accession}: {e}")
                result.metadata["curation_error"] = str(e)
                return (source_name, idx, result, False)
        
        # Process in parallel with ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=CURATION_WORKERS) as executor:
            # Submit all items
            future_to_item = {
                executor.submit(curate_single, item): item 
                for item in items_to_curate
            }
            
            # Process completed futures
            for future in as_completed(future_to_item):
                # Check runtime limit
                if time.time() - start_time > MAX_RUNTIME_SECONDS - 60:
                    logger.warning("Runtime limit approaching, cancelling remaining curation tasks")
                    executor.shutdown(wait=False, cancel_futures=True)
                    break
                
                try:
                    source_name, idx, curated_result, success = future.result()
                    # Update the result in all_results
                    all_results[source_name][idx] = curated_result
                    
                    if success:
                        curated_count += 1
                    else:
                        failed_count += 1
                    
                    # Progress logging
                    total_processed = curated_count + failed_count
                    if total_processed % 50 == 0:
                        elapsed = time.time() - curation_start
                        rate = total_processed / elapsed * 60 if elapsed > 0 else 0
                        logger.info(
                            f"Curation progress: {curated_count}/{effective_limit} success, "
                            f"{failed_count} failed ({rate:.1f}/min) [workers={CURATION_WORKERS}]"
                        )
                except Exception as e:
                    failed_count += 1
                    logger.warning(f"Curation future failed: {e}")
        
        skipped_count = total_uncurated - effective_limit
        curation_elapsed = time.time() - curation_start
        final_rate = (curated_count + failed_count) / curation_elapsed * 60 if curation_elapsed > 0 else 0
        logger.info(
            f"Curation complete: {curated_count} success, {failed_count} failed, "
            f"{skipped_count} skipped ({final_rate:.1f}/min)"
        )
    
    # Summary statistics
    total_results = sum(len(r) for r in all_results.values())
    elapsed = time.time() - start_time
    logger.info(f"Harvesting complete: {total_results} datasets in {elapsed:.1f}s")
    
    return all_results


# --- CLOUD FUNCTIONS HANDLERS ---

@functions_framework.http
def scan_geo(request):
    """
    Main HTTP handler for GEO/multi-source scanning.
    
    Query parameters:
    - modalities: Comma-separated list of modality sources to harvest
    - max_per_modality: Maximum results per modality (default: 100)
    - enable_curation: Whether to run Gemini curation (default: true)
    - max_curations: Maximum items to curate per run (default: env MAX_CURATIONS_PER_RUN)
    - save_results: Whether to save results to GCS (default: true)
    - disease_focus: Disease focus area (default: both)
    
    Returns:
        JSON response with harvest summary and results
    """
    try:
        # Parse request parameters
        request_json = request.get_json(silent=True) or {}
        request_args = request.args
        
        modalities_str = request_args.get("modalities") or request_json.get("modalities")
        modalities = modalities_str.split(",") if modalities_str else None
        
        max_per_modality = int(
            request_args.get("max_per_modality") or 
            request_json.get("max_per_modality", 100)
        )
        
        enable_curation = str(
            request_args.get("enable_curation") or 
            request_json.get("enable_curation", "true")
        ).lower() == "true"
        
        max_curations = int(
            request_args.get("max_curations") or 
            request_json.get("max_curations", 0)
        )
        
        save_results = str(
            request_args.get("save_results") or 
            request_json.get("save_results", "true")
        ).lower() == "true"
        
        logger.info(f"Starting harvest: modalities={modalities}, max={max_per_modality}, "
                   f"curation={enable_curation}, max_curations={max_curations}, save={save_results}")
        
        # Execute harvesting
        all_results = harvest_all_modalities(
            max_per_modality=max_per_modality,
            enable_curation=enable_curation,
            modalities=modalities,
            max_curations=max_curations
        )
        
        # Save results to GCS
        saved_count = 0
        if save_results:
            for source_name, results in all_results.items():
                for result in results:
                    if save_result(result):
                        saved_count += 1
        
        # Build response summary
        total_datasets = sum(len(r) for r in all_results.values())
        total_curated = sum(1 for results in all_results.values() for r in results if r.curated)
        total_uncurated = total_datasets - total_curated
        
        summary = {
            "status": "success",
            "timestamp": datetime.utcnow().isoformat(),
            "sources": {},
            "totals": {
                "datasets": total_datasets,
                "curated": total_curated,
                "uncurated": total_uncurated,
                "saved": saved_count,
            },
            "config": {
                "max_per_modality": max_per_modality,
                "enable_curation": enable_curation,
                "curation_provider": CURATION_PROVIDER,
                "curation_model": CLAUDE_MODEL if CURATION_PROVIDER == "claude" else GEMINI_MODEL,
                "curation_region": CLAUDE_REGION if CURATION_PROVIDER == "claude" else VERTEX_LOCATION,
                "curation_workers": CURATION_WORKERS,
                "max_curations": max_curations if max_curations > 0 else MAX_CURATIONS_PER_RUN,
                "curation_delay_seconds": CURATION_DELAY_SECONDS,
            },
            "modality_breakdown": {},
        }
        
        # Per-source summary
        for source_name, results in all_results.items():
            summary["sources"][source_name] = {
                "count": len(results),
                "curated": sum(1 for r in results if r.curated),
                "modalities": list(set(r.modality for r in results)),
            }
        
        # Modality breakdown
        modality_counts: Dict[str, int] = {}
        for results in all_results.values():
            for r in results:
                modality_counts[r.modality] = modality_counts.get(r.modality, 0) + 1
        summary["modality_breakdown"] = modality_counts
        
        # Include sample results (first few from each source)
        summary["sample_results"] = {}
        for source_name, results in all_results.items():
            summary["sample_results"][source_name] = [
                {
                    "accession": r.accession,
                    "title": r.title[:100],
                    "modality": r.modality,
                    "curated": r.curated,
                    "graph_functions": r.graph_functions[:3] if r.graph_functions else [],
                }
                for r in results[:3]
            ]
        
        return json.dumps(summary, indent=2), 200, {"Content-Type": "application/json"}
    
    except Exception as e:
        logger.error(f"Handler error: {e}")
        logger.debug(traceback.format_exc())
        
        error_response = {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat(),
        }
        return json.dumps(error_response), 500, {"Content-Type": "application/json"}


@functions_framework.http  
def harvest_specific_modality(request):
    """
    Handler to harvest a specific modality.
    
    Query parameters:
    - modality: The modality to harvest (e.g., "scrna", "proteomics", "flow")
    - max_results: Maximum results (default: 100)
    - disease_query: Custom disease query string
    """
    try:
        request_json = request.get_json(silent=True) or {}
        request_args = request.args
        
        modality_str = (
            request_args.get("modality") or 
            request_json.get("modality", "bulk_rna")
        ).lower()
        
        max_results = int(
            request_args.get("max_results") or 
            request_json.get("max_results", 100)
        )
        
        disease_query = (
            request_args.get("disease_query") or 
            request_json.get("disease_query", "trained immunity OR kidney transplant")
        )
        
        # Map string to Modality enum
        try:
            modality = Modality(modality_str)
        except ValueError:
            return json.dumps({
                "status": "error",
                "error": f"Unknown modality: {modality_str}",
                "valid_modalities": [m.value for m in Modality],
            }), 400, {"Content-Type": "application/json"}
        
        logger.info(f"Harvesting {modality.value} with max={max_results}")
        
        # Execute modality-specific harvest
        results = harvest_geo_modality(
            modality=modality,
            disease_context=disease_query,
            max_results=max_results
        )
        
        # Save results
        saved_count = 0
        for result in results:
            if save_result(result):
                saved_count += 1
        
        response = {
            "status": "success",
            "modality": modality.value,
            "count": len(results),
            "saved": saved_count,
            "timestamp": datetime.utcnow().isoformat(),
            "results": [
                {
                    "accession": r.accession,
                    "title": r.title[:200],
                    "summary": r.summary[:500],
                    "organism": r.organism,
                    "sample_count": r.sample_count,
                }
                for r in results[:20]  # Return first 20 as preview
            ]
        }
        
        return json.dumps(response, indent=2), 200, {"Content-Type": "application/json"}
    
    except Exception as e:
        logger.error(f"Modality harvest error: {e}")
        return json.dumps({
            "status": "error",
            "error": str(e),
        }), 500, {"Content-Type": "application/json"}


@functions_framework.http
def check_harvest_status(request):
    """
    Handler to check harvesting status and statistics.
    """
    try:
        client = get_storage_client()
        
        if not client:
            return json.dumps({
                "status": "local_mode",
                "message": "Running without GCS - no persistent state available",
            }), 200, {"Content-Type": "application/json"}
        
        bucket = client.bucket(BUCKET_NAME)
        
        # Count results by source
        stats = {}
        blobs = bucket.list_blobs(prefix="harvest_results/")
        
        for blob in blobs:
            parts = blob.name.split("/")
            if len(parts) >= 2:
                source = parts[1]
                stats[source] = stats.get(source, 0) + 1
        
        # Get state files
        states = {}
        state_blobs = bucket.list_blobs(prefix="harvest_state/")
        for blob in state_blobs:
            if blob.name.endswith("_state.json"):
                try:
                    data = json.loads(blob.download_as_text())
                    source = data.get("source", "unknown")
                    states[source] = {
                        "total_processed": data.get("total_processed", 0),
                        "last_updated": data.get("updated_at"),
                        "completed": data.get("completed", False),
                    }
                except Exception:
                    pass
        
        response = {
            "status": "ok",
            "bucket": BUCKET_NAME,
            "results_by_source": stats,
            "total_results": sum(stats.values()),
            "processing_states": states,
            "timestamp": datetime.utcnow().isoformat(),
        }
        
        return json.dumps(response, indent=2), 200, {"Content-Type": "application/json"}
    
    except Exception as e:
        logger.error(f"Status check error: {e}")
        return json.dumps({
            "status": "error",
            "error": str(e),
        }), 500, {"Content-Type": "application/json"}


@functions_framework.http
def curate_pending(request):
    """
    Handler to run Gemini curation on pending (uncurated) results.
    """
    try:
        request_json = request.get_json(silent=True) or {}
        
        max_to_curate = int(request_json.get("max_to_curate", 50))
        source_filter = request_json.get("source")
        
        client = get_storage_client()
        if not client:
            return json.dumps({
                "status": "error",
                "error": "GCS not available",
            }), 500, {"Content-Type": "application/json"}
        
        bucket = client.bucket(BUCKET_NAME)
        
        # Find uncurated results
        prefix = f"harvest_results/{source_filter}/" if source_filter else "harvest_results/"
        blobs = bucket.list_blobs(prefix=prefix)
        
        curated_count = 0
        errors = []
        
        for blob in blobs:
            if curated_count >= max_to_curate:
                break
            
            if not blob.name.endswith(".json"):
                continue
            
            try:
                data = json.loads(blob.download_as_text())
                
                # Skip if already curated
                if data.get("curated", False):
                    continue
                
                # Reconstruct HarvestResult
                result = HarvestResult(**data)
                
                # Curate using configured provider
                curated_result = curate_dataset(result)
                
                # Save back
                blob.upload_from_string(
                    json.dumps(asdict(curated_result)),
                    content_type="application/json"
                )
                
                curated_count += 1
                
                if curated_count % 10 == 0:
                    logger.info(f"Curated {curated_count} datasets...")
                
            except Exception as e:
                errors.append({"blob": blob.name, "error": str(e)})
                continue
        
        response = {
            "status": "success",
            "curated_count": curated_count,
            "errors": errors[:10],  # First 10 errors
            "timestamp": datetime.utcnow().isoformat(),
        }
        
        return json.dumps(response, indent=2), 200, {"Content-Type": "application/json"}
    
    except Exception as e:
        logger.error(f"Curation error: {e}")
        return json.dumps({
            "status": "error",
            "error": str(e),
        }), 500, {"Content-Type": "application/json"}


# --- HARVEST STATE MANAGEMENT (for auto-continuation) ---

def get_harvest_state_path() -> str:
    """Get the GCS path for harvest state file."""
    return f"harvest_state/current_state.json"


def get_seen_accessions_path() -> str:
    """Get the GCS path for seen accessions file."""
    return f"harvest_state/seen_accessions.json"


def load_seen_accessions() -> Set[str]:
    """
    Load set of already-harvested accession IDs from GCS.
    
    Returns:
        Set of accession IDs that have been previously harvested
    """
    try:
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(get_seen_accessions_path())
        
        if blob.exists():
            content = blob.download_as_string()
            data = json.loads(content)
            return set(data.get("accessions", []))
    except Exception as e:
        logger.warning(f"Could not load seen accessions: {e}")
    
    return set()


def save_seen_accessions(accessions: Set[str]) -> bool:
    """Save set of seen accessions to GCS."""
    try:
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(get_seen_accessions_path())
        
        data = {
            "accessions": list(accessions),
            "count": len(accessions),
            "updated_at": datetime.utcnow().isoformat(),
        }
        blob.upload_from_string(
            json.dumps(data),
            content_type="application/json"
        )
        return True
    except Exception as e:
        logger.error(f"Could not save seen accessions: {e}")
        return False


def add_seen_accessions(new_accessions: List[str]) -> int:
    """
    Add new accessions to the seen set (incremental update).
    
    Returns:
        Number of newly added accessions
    """
    existing = load_seen_accessions()
    new_set = set(new_accessions)
    truly_new = new_set - existing
    
    if truly_new:
        updated = existing | truly_new
        save_seen_accessions(updated)
        logger.info(f"Added {len(truly_new)} new accessions to seen set (total: {len(updated)})")
    
    return len(truly_new)


def deduplicate_gcs_data() -> Dict[str, Any]:
    """
    Scan GCS bucket and remove duplicate files, keeping only the most recent.
    
    Returns:
        Summary of deduplication results
    """
    try:
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        
        # Scan all harvest results
        prefix = "harvest_results/"
        blobs = list(bucket.list_blobs(prefix=prefix))
        
        logger.info(f"Scanning {len(blobs)} files for duplicates...")
        
        # Group by accession
        accession_files: Dict[str, List[Tuple[str, datetime]]] = {}
        
        for blob in blobs:
            if not blob.name.endswith('.json'):
                continue
            
            # Extract accession from path: harvest_results/{source}/{date}/{accession}.json
            parts = blob.name.split('/')
            if len(parts) >= 4:
                accession = parts[-1].replace('.json', '')
                if accession not in accession_files:
                    accession_files[accession] = []
                accession_files[accession].append((blob.name, blob.updated))
        
        # Find duplicates and delete older versions
        duplicates_found = 0
        files_deleted = 0
        unique_accessions = set()
        
        for accession, files in accession_files.items():
            unique_accessions.add(accession)
            
            if len(files) > 1:
                duplicates_found += 1
                # Sort by date, keep newest
                files.sort(key=lambda x: x[1], reverse=True)
                
                # Delete all but the newest
                for filepath, _ in files[1:]:
                    try:
                        bucket.blob(filepath).delete()
                        files_deleted += 1
                        logger.debug(f"Deleted duplicate: {filepath}")
                    except Exception as e:
                        logger.warning(f"Could not delete {filepath}: {e}")
        
        # Update seen accessions with all unique ones
        save_seen_accessions(unique_accessions)
        
        result = {
            "total_files_scanned": len(blobs),
            "unique_accessions": len(unique_accessions),
            "duplicates_found": duplicates_found,
            "files_deleted": files_deleted,
            "timestamp": datetime.utcnow().isoformat(),
        }
        
        logger.info(f"Deduplication complete: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Deduplication error: {e}")
        return {"error": str(e)}


def load_harvest_state() -> Dict[str, Any]:
    """
    Load the current harvest state from GCS.
    
    Returns:
        Dictionary with harvest state including:
        - completed_sources: List of completed source names
        - page_offsets: Dict of source -> current page/offset
        - total_harvested: Total count per source
        - last_run: Timestamp of last run
        - exhausted: Whether harvesting is complete
    """
    try:
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(get_harvest_state_path())
        
        if blob.exists():
            content = blob.download_as_string()
            return json.loads(content)
    except Exception as e:
        logger.warning(f"Could not load harvest state: {e}")
    
    # Return default state
    return {
        "completed_sources": [],
        "page_offsets": {},
        "total_harvested": {},
        "last_run": None,
        "exhausted": False,
        "session_id": datetime.utcnow().strftime("%Y%m%d_%H%M%S"),
    }


def save_harvest_state(state: Dict[str, Any]) -> bool:
    """Save harvest state to GCS."""
    try:
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(get_harvest_state_path())
        
        state["last_run"] = datetime.utcnow().isoformat()
        blob.upload_from_string(
            json.dumps(state, indent=2),
            content_type="application/json"
        )
        return True
    except Exception as e:
        logger.error(f"Could not save harvest state: {e}")
        return False


def reset_harvest_state() -> Dict[str, Any]:
    """Reset harvest state to start fresh."""
    state = {
        "completed_sources": [],
        "page_offsets": {},
        "total_harvested": {},
        "last_run": None,
        "exhausted": False,
        "session_id": datetime.utcnow().strftime("%Y%m%d_%H%M%S"),
    }
    save_harvest_state(state)
    return state


def get_paginated_harvester(source_name: str, max_results: int, offset: int, 
                            seen_accessions: Set[str]) -> Tuple[List[HarvestResult], int, bool]:
    """
    Get a paginated harvester for a specific source.
    
    Returns:
        Tuple of (results, next_offset, is_exhausted)
    """
    # Map source names to modalities and queries
    source_to_modality = {
        "geo_bulk_rna": Modality.BULK_RNA,
        "geo_scrna": Modality.SCRNA,
        "geo_mirna": Modality.MIRNA,
        "geo_longread": Modality.LONGREAD,
        "geo_wgs": Modality.WGS,
        "geo_wes": Modality.WES,
        "geo_hla": Modality.HLA,
        "geo_snp": Modality.SNP,
        "geo_chipseq": Modality.CHIPSEQ,
        "geo_atac": Modality.ATAC,
        "geo_cytokine": Modality.CYTOKINE,
    }
    
    # Handle modality-based sources (use harvest_ncbi_deep_paginated)
    if source_name in source_to_modality:
        modality = source_to_modality[source_name]
        keywords = MODALITY_QUERIES.get(modality, [])
        keyword_query = " OR ".join(f'"{kw}"' for kw in keywords[:5])
        query = f"({keyword_query}) AND (trained immunity OR kidney transplant) AND Homo sapiens[Organism]"
        
        return harvest_ncbi_deep_paginated(
            db="gds",
            query=query,
            modality_hint=modality,
            max_results=max_results,
            retstart_offset=offset,
            seen_accessions=seen_accessions
        )
    
    # Non-paginated sources - use regular harvesters but filter seen accessions
    # These don't have true pagination support yet
    if source_name == "spatial":
        results = harvest_spatial_transcriptomics(max_results=max_results)
    elif source_name == "eccdna":
        results = harvest_eccdna(max_results=max_results)
    elif source_name == "cnv":
        results = harvest_cnv(max_results=max_results)
    elif source_name == "methylation":
        results = harvest_methylation(max_results=max_results)
    elif source_name == "hic":
        results = harvest_hic(max_results=max_results)
    elif source_name == "glycomics":
        results = harvest_glycomics(max_results=max_results)
    elif source_name == "metabolomics":
        results = harvest_metabolomics(max_results=max_results)
    elif source_name == "lipidomics":
        results = harvest_lipidomics(max_results=max_results)
    elif source_name == "fluxomics":
        results = harvest_fluxomics(max_results=max_results)
    elif source_name == "immport":
        results = harvest_immport(max_results=max_results)
    elif source_name == "flowrepo":
        results = harvest_flowrepo(max_results=max_results)
    elif source_name == "cytof":
        results = harvest_cytof(max_results=max_results)
    elif source_name == "multiplex_imaging":
        results = harvest_multiplex_imaging(max_results=max_results)
    elif source_name == "functional_assays":
        results = harvest_functional_assays(max_results=max_results)
    elif source_name == "repertoire":
        results = harvest_repertoire(max_results=max_results)
    elif source_name == "microbiome":
        results = harvest_microbiome(max_results=max_results)
    elif source_name == "citeseq_multiome":
        results = harvest_citeseq_multiome(max_results=max_results)
    elif source_name == "perturbseq":
        results = harvest_perturbseq(max_results=max_results)
    elif source_name == "pride":
        results = harvest_pride(max_results=max_results)
    elif source_name == "arrayexpress":
        results = harvest_arrayexpress(max_results=max_results)
    elif source_name == "kpmp":
        results = harvest_kpmp(max_results=max_results)
    elif source_name == "hca":
        results = harvest_hca_kidney(max_results=max_results)
    else:
        return [], offset, True
    
    # Filter out already-seen accessions
    new_results = [r for r in results if r.accession not in seen_accessions]
    
    # For non-paginated sources, we mark as exhausted after first run
    # (these sources don't support true pagination)
    is_exhausted = True
    
    return new_results, offset + len(results), is_exhausted


@functions_framework.http
def harvest_exhaustive(request):
    """
    Continuous harvesting endpoint with pagination and deduplication.
    
    Tracks seen accessions to avoid duplicates. Uses NCBI pagination for supported sources.
    Designed to be called repeatedly until all data is exhausted.
    
    Query parameters:
    - reset: If "true", reset state and start fresh (also clears seen accessions)
    - max_per_source: Maximum results per source per run (default: 200)
    - max_curations: Maximum curations per run (default: 500)
    - dedupe_first: If "true", run deduplication before harvesting (default: false)
    
    Returns:
        JSON with progress and harvested counts
    """
    try:
        request_json = request.get_json(silent=True) or {}
        request_args = request.args
        
        # Parse parameters
        reset = str(request_args.get("reset") or request_json.get("reset", "false")).lower() == "true"
        max_per_source = int(request_args.get("max_per_source") or request_json.get("max_per_source", 200))
        max_curations = int(request_args.get("max_curations") or request_json.get("max_curations", 500))
        dedupe_first = str(request_args.get("dedupe_first") or request_json.get("dedupe_first", "false")).lower() == "true"
        
        # Optionally run deduplication first
        dedupe_result = None
        if dedupe_first:
            logger.info("Running deduplication before harvest...")
            dedupe_result = deduplicate_gcs_data()
        
        # Load or reset state
        if reset:
            state = reset_harvest_state()
            # Also clear seen accessions on reset
            save_seen_accessions(set())
            logger.info("Harvest state reset - starting fresh (cleared seen accessions)")
        else:
            state = load_harvest_state()
        
        # Check if already exhausted
        if state.get("exhausted"):
            return json.dumps({
                "status": "exhausted",
                "message": "All sources have been fully harvested. Use ?reset=true to start over.",
                "progress": state,
                "unique_accessions": len(load_seen_accessions()),
            }, indent=2), 200, {"Content-Type": "application/json"}
        
        # Load seen accessions to avoid duplicates
        seen_accessions = load_seen_accessions()
        logger.info(f"Loaded {len(seen_accessions)} previously seen accessions")
        
        # Define all harvestable sources
        all_sources = [
            # Transcriptomics (paginated)
            "geo_bulk_rna", "geo_scrna", "geo_mirna", "geo_longread",
            # Genomics (paginated)
            "geo_wgs", "geo_wes", "geo_hla", "geo_snp",
            # Epigenomics (paginated)
            "geo_chipseq", "geo_atac",
            # Proteomics (paginated)
            "geo_cytokine",
            # Non-paginated sources (run once each)
            "spatial", "eccdna", "cnv", "methylation", "hic",
            "glycomics", "metabolomics", "lipidomics", "fluxomics",
            "immport", "flowrepo", "cytof", "multiplex_imaging", "functional_assays",
            "repertoire", "microbiome", "citeseq_multiome", "perturbseq",
            "pride", "arrayexpress", "kpmp", "hca",
        ]
        
        # Find sources not yet completed
        completed = set(state.get("completed_sources", []))
        remaining_sources = [s for s in all_sources if s not in completed]
        
        if not remaining_sources:
            state["exhausted"] = True
            save_harvest_state(state)
            return json.dumps({
                "status": "exhausted",
                "message": "All sources have been fully harvested!",
                "progress": state,
                "unique_accessions": len(seen_accessions),
            }, indent=2), 200, {"Content-Type": "application/json"}
        
        start_time = time.time()
        harvested_this_run: Dict[str, int] = {}
        new_accessions_this_run: List[str] = []
        newly_completed = []
        
        # Process sources (limit to avoid timeout)
        sources_to_process = remaining_sources[:5]  # Process 5 sources per run for better progress
        
        for source_name in sources_to_process:
            # Check runtime
            if time.time() - start_time > MAX_RUNTIME_SECONDS - 300:  # 5 min buffer
                logger.warning("Approaching runtime limit, saving progress")
                break
            
            # Get current offset for this source
            offset = state.get("page_offsets", {}).get(source_name, 0)
            
            logger.info(f"Harvesting {source_name} from offset {offset} (seen: {len(seen_accessions)})")
            
            # Harvest this source with pagination and deduplication
            try:
                results, next_offset, is_exhausted = get_paginated_harvester(
                    source_name=source_name,
                    max_results=max_per_source,
                    offset=offset,
                    seen_accessions=seen_accessions
                )
                
                count = len(results)
                harvested_this_run[source_name] = count
                
                # Track new accessions
                for r in results:
                    new_accessions_this_run.append(r.accession)
                    seen_accessions.add(r.accession)
                
                # Update totals
                if "total_harvested" not in state:
                    state["total_harvested"] = {}
                state["total_harvested"][source_name] = state["total_harvested"].get(source_name, 0) + count
                
                # Update offset
                if "page_offsets" not in state:
                    state["page_offsets"] = {}
                state["page_offsets"][source_name] = next_offset
                
                # Run AI curation on new results
                curated_count = 0
                for result in results[:max_curations // len(sources_to_process)]:
                    try:
                        curated = curate_dataset(result)
                        save_result(curated)
                        curated_count += 1
                    except Exception as e:
                        logger.warning(f"Curation failed for {result.accession}: {e}")
                        save_result(result)
                
                # Save remaining uncurated results
                for result in results[max_curations // len(sources_to_process):]:
                    save_result(result)
                
                logger.info(f"Source {source_name}: {count} new, {curated_count} curated, offset now {next_offset}")
                
                # Mark source as completed if exhausted
                if is_exhausted:
                    state["completed_sources"].append(source_name)
                    newly_completed.append(source_name)
                    logger.info(f"Source {source_name} COMPLETED (exhausted)")
                    
            except Exception as e:
                logger.error(f"Error harvesting {source_name}: {e}")
                logger.debug(traceback.format_exc())
                harvested_this_run[source_name] = 0
        
        # Save seen accessions incrementally
        if new_accessions_this_run:
            add_seen_accessions(new_accessions_this_run)
        
        # Save state
        save_harvest_state(state)
        
        # Calculate progress
        total_sources = len(all_sources)
        completed_sources = len(state.get("completed_sources", []))
        progress_pct = (completed_sources / total_sources) * 100 if total_sources > 0 else 0
        
        remaining = [s for s in all_sources if s not in state.get("completed_sources", [])]
        
        response = {
            "status": "in_progress" if remaining else "exhausted",
            "session_id": state.get("session_id"),
            "progress": {
                "completed_sources": completed_sources,
                "total_sources": total_sources,
                "percentage": round(progress_pct, 1),
                "total_datasets": sum(state.get("total_harvested", {}).values()),
                "unique_accessions": len(seen_accessions),
            },
            "harvested_this_run": harvested_this_run,
            "new_accessions_this_run": len(new_accessions_this_run),
            "newly_completed": newly_completed,
            "remaining_sources": remaining[:10],
            "config": {
                "max_per_source": max_per_source,
                "max_curations": max_curations,
                "curation_provider": CURATION_PROVIDER,
            },
        }
        
        if dedupe_result:
            response["deduplication"] = dedupe_result
        
        # Mark as exhausted if no remaining
        if not remaining:
            state["exhausted"] = True
            save_harvest_state(state)
            response["status"] = "exhausted"
            response["message"] = "All sources have been fully harvested!"
        
        return json.dumps(response, indent=2), 200, {"Content-Type": "application/json"}
        
    except Exception as e:
        logger.error(f"Exhaustive harvest error: {e}")
        logger.debug(traceback.format_exc())
        
        return json.dumps({
            "status": "error",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat(),
        }), 500, {"Content-Type": "application/json"}


@functions_framework.http
def get_harvest_progress(request):
    """
    Get current harvest progress without running any harvesting.
    
    Returns the current state of the exhaustive harvest.
    """
    try:
        state = load_harvest_state()
        
        # Define all sources for percentage calculation
        all_sources = [
            "geo_bulk_rna", "geo_scrna", "spatial", "geo_mirna", "geo_longread", "eccdna",
            "geo_wgs", "geo_wes", "geo_hla", "geo_snp", "cnv",
            "geo_chipseq", "geo_atac", "methylation", "hic",
            "geo_cytokine", "glycomics",
            "metabolomics", "lipidomics", "fluxomics",
            "immport", "flowrepo", "cytof", "multiplex_imaging", "functional_assays",
            "repertoire", "microbiome",
            "citeseq_multiome", "perturbseq",
            "pride", "arrayexpress", "kpmp", "hca",
        ]
        
        completed = state.get("completed_sources", [])
        remaining = [s for s in all_sources if s not in completed]
        
        progress_pct = (len(completed) / len(all_sources)) * 100 if all_sources else 0
        
        # Load unique accessions count
        seen_count = len(load_seen_accessions())
        
        return json.dumps({
            "status": "exhausted" if state.get("exhausted") else ("idle" if not state.get("last_run") else "in_progress"),
            "session_id": state.get("session_id"),
            "last_run": state.get("last_run"),
            "progress": {
                "completed_sources": len(completed),
                "total_sources": len(all_sources),
                "percentage": round(progress_pct, 1),
                "total_datasets": sum(state.get("total_harvested", {}).values()),
                "unique_accessions": seen_count,
            },
            "completed_sources": completed,
            "remaining_sources": remaining,
            "datasets_per_source": state.get("total_harvested", {}),
        }, indent=2), 200, {"Content-Type": "application/json"}
        
    except Exception as e:
        return json.dumps({
            "status": "error",
            "error": str(e),
        }), 500, {"Content-Type": "application/json"}


@functions_framework.http
def deduplicate_data(request):
    """
    Deduplicate harvested data in GCS.
    
    Scans all harvest results, removes duplicate files (keeping newest),
    and updates the seen accessions set.
    
    Query parameters:
    - dry_run: If "true", only report what would be deleted (default: false)
    
    Returns:
        JSON with deduplication statistics
    """
    try:
        request_args = request.args
        dry_run = str(request_args.get("dry_run", "false")).lower() == "true"
        
        if dry_run:
            # Just count duplicates without deleting
            try:
                client = storage.Client()
                bucket = client.bucket(BUCKET_NAME)
                
                prefix = "harvest_results/"
                blobs = list(bucket.list_blobs(prefix=prefix))
                
                accession_counts: Dict[str, int] = {}
                for blob in blobs:
                    if blob.name.endswith('.json'):
                        parts = blob.name.split('/')
                        if len(parts) >= 4:
                            accession = parts[-1].replace('.json', '')
                            accession_counts[accession] = accession_counts.get(accession, 0) + 1
                
                duplicates = {k: v for k, v in accession_counts.items() if v > 1}
                
                return json.dumps({
                    "status": "dry_run",
                    "total_files": len(blobs),
                    "unique_accessions": len(accession_counts),
                    "accessions_with_duplicates": len(duplicates),
                    "total_duplicate_files": sum(v - 1 for v in duplicates.values()),
                    "sample_duplicates": dict(list(duplicates.items())[:10]),
                }, indent=2), 200, {"Content-Type": "application/json"}
                
            except Exception as e:
                return json.dumps({"error": str(e)}), 500, {"Content-Type": "application/json"}
        
        # Actually run deduplication
        result = deduplicate_gcs_data()
        
        return json.dumps({
            "status": "completed",
            **result,
        }, indent=2), 200, {"Content-Type": "application/json"}
        
    except Exception as e:
        logger.error(f"Deduplication error: {e}")
        return json.dumps({
            "status": "error",
            "error": str(e),
        }), 500, {"Content-Type": "application/json"}


# --- FILE DOWNLOAD ENDPOINTS ---

@functions_framework.http
def download_files(request):
    """
    HTTP endpoint to download data files for harvested datasets.
    """
    try:
        # 1. Parse Arguments
        request_args = request.args if hasattr(request, "args") else {}
        
        accession = request_args.get("accession")
        source = request_args.get("source")
        max_files = int(request_args.get("max_files", "5"))
        max_size_mb = int(request_args.get("max_size_mb", "500"))
        dry_run = str(request_args.get("dry_run", "false")).lower() == "true"
        limit = int(request_args.get("limit", "10"))
        
        results = []
        
        # 2. Logic Flow
        if accession:
            # Single dataset logic
            files = discover_dataset_files(accession)
            # Filter for priority
            downloaded = [] 
            if not dry_run:
                downloaded = download_dataset_files(accession, "bulk_rna", max_files=max_files, max_size_mb=max_size_mb)
            
            results = [{
                "accession": accession,
                "files_found": len(files),
                "files_downloaded": sum(1 for f in downloaded if f.downloaded),
                "files": [
                    {
                        "filename": f.filename,
                        "downloaded": f.downloaded,
                        "gcs_path": f.gcs_path,
                        "priority_score": f.priority_score,
                    }
                    for f in downloaded
                ],
            }]

        elif source:
            # Batch mode logic
            # NOTE: Logic specific to batch downloading (e.g. calling download_pending_files) goes here.
            # Returning empty results for now to satisfy syntax.
            return json.dumps({"status": "success", "results": results}), 200, {"Content-Type": "application/json"}

        else:
            # [FIX]: This else is now correctly aligned with 'if' and 'elif'
            return json.dumps({
                "status": "error",
                "error": "Must specify either 'accession' or 'source' parameter",
                "usage": {
                    "single": "?accession=GSE12345&modality=scrna&max_files=5",
                    "batch": "?source=geo_scrna&limit=10",
                    "dry_run": "Add &dry_run=true to preview without downloading",
                },
            }), 400, {"Content-Type": "application/json"}
        
        # 3. Final Return (for accession path)
        total_downloaded = sum(r.get("files_downloaded", 0) for r in results)
        
        return json.dumps({
            "status": "success",
            "datasets_processed": len(results),
            "total_files_downloaded": total_downloaded,
            "results": results,
        }, indent=2), 200, {"Content-Type": "application/json"}
        
    except Exception as e:
        # [FIX]: Consolidated error handling
        logger.error(f"Download files error: {e}")
        logger.debug(traceback.format_exc())
        return json.dumps({
            "status": "error",
            "error": str(e),
        }), 500, {"Content-Type": "application/json"}


@functions_framework.http
def list_data_files(request):
    """
    HTTP endpoint to list downloaded data files.
    
    Query Parameters:
        accession: Specific accession to check
        prefix: GCS prefix to list (default: "data_files/")
        limit: Maximum results (default: 100)
    
    Example:
        ?accession=GSE12345
        ?prefix=data_files/&limit=50
    """
    try:
        request_args = request.args if hasattr(request, "args") else {}
        
        accession = request_args.get("accession")
        prefix = request_args.get("prefix", "data_files/")
        limit = int(request_args.get("limit", "100"))
        
        client = get_storage_client()
        if not client:
            return json.dumps({
                "status": "error",
                "error": "Storage client not available",
            }), 500, {"Content-Type": "application/json"}
        
        bucket = client.bucket(BUCKET_NAME)
        
        if accession:
            # Get files for specific accession
            info = get_dataset_file_info(accession)
            return json.dumps({
                "status": "success",
                "accession": accession,
                **info,
            }, indent=2), 200, {"Content-Type": "application/json"}
        
        # List all files under prefix
        blobs = list(bucket.list_blobs(prefix=prefix, max_results=limit))
        
        # Group by accession
        by_accession = {}
        total_size = 0
        
        for blob in blobs:
            parts = blob.name.split('/')
            if len(parts) >= 3:
                acc = parts[1]
                if acc not in by_accession:
                    by_accession[acc] = {"files": [], "size": 0}
                by_accession[acc]["files"].append(parts[-1])
                by_accession[acc]["size"] += blob.size or 0
                total_size += blob.size or 0
        
        return json.dumps({
            "status": "success",
            "total_files": len(blobs),
            "total_size_mb": round(total_size / 1024 / 1024, 2),
            "datasets_with_files": len(by_accession),
            "datasets": {
                k: {
                    "file_count": len(v["files"]),
                    "size_mb": round(v["size"] / 1024 / 1024, 2),
                    "files": v["files"][:5],  # Show first 5
                }
                for k, v in list(by_accession.items())[:50]
            },
        }, indent=2), 200, {"Content-Type": "application/json"}
        
    except Exception as e:
        logger.error(f"List data files error: {e}")
        return json.dumps({
            "status": "error",
            "error": str(e),
        }), 500, {"Content-Type": "application/json"}


def load_attempted_downloads() -> Set[str]:
    """Load set of accessions that have been attempted (with or without success)."""
    client = get_storage_client()
    if not client:
        return set()
    
    try:
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob("download_state/attempted_accessions.json")
        if blob.exists():
            data = json.loads(blob.download_as_string())
            return set(data.get("accessions", []))
    except Exception as e:
        logger.debug(f"Could not load attempted downloads: {e}")
    
    return set()


def save_attempted_downloads(accessions: Set[str]):
    """Save set of attempted accessions."""
    client = get_storage_client()
    if not client:
        return
    
    try:
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob("download_state/attempted_accessions.json")
        blob.upload_from_string(
            json.dumps({"accessions": list(accessions), "updated": datetime.utcnow().isoformat()}),
            content_type="application/json"
        )
    except Exception as e:
        logger.warning(f"Could not save attempted downloads: {e}")


def add_attempted_downloads(new_accessions: List[str]):
    """Add accessions to the attempted set."""
    existing = load_attempted_downloads()
    existing.update(new_accessions)
    save_attempted_downloads(existing)


@functions_framework.http
def download_pending_files(request):
    """
    HTTP endpoint to download files for datasets that don't have files yet.
    
    This endpoint scans harvested datasets and downloads data files for those
    that don't already have files in GCS.
    
    Query Parameters:
        source: Limit to specific source (optional)
        modality: Limit to specific modality (optional)
        limit: Maximum datasets to process (default: 20)
        max_files: Maximum files per dataset (default: 3)
        max_size_mb: Maximum file size (default: 200)
        dry_run: Preview without downloading (default: false)
        reset_attempted: Reset the attempted tracking (default: false)
        geo_first: Process GEO datasets before others (default: true)
    
    Example:
        ?source=geo_scrna&limit=50
        ?modality=scrna&limit=100&dry_run=true
    """
    try:
        request_args = request.args if hasattr(request, "args") else {}
        
        source_filter = request_args.get("source")
        modality_filter = request_args.get("modality")
        limit = int(request_args.get("limit", "20"))
        max_files = int(request_args.get("max_files", "3"))
        max_size_mb = int(request_args.get("max_size_mb", "200"))
        dry_run = str(request_args.get("dry_run", "false")).lower() == "true"
        reset_attempted = str(request_args.get("reset_attempted", "false")).lower() == "true"
        geo_first = str(request_args.get("geo_first", "true")).lower() == "true"
        
        client = get_storage_client()
        if not client:
            return json.dumps({
                "status": "error",
                "error": "Storage client not available",
            }), 500, {"Content-Type": "application/json"}
        
        bucket = client.bucket(BUCKET_NAME)
        
        # Reset attempted tracking if requested
        if reset_attempted:
            save_attempted_downloads(set())
            logger.info("Reset attempted downloads tracking")
        
        # Load already-attempted accessions
        attempted_accessions = load_attempted_downloads()
        logger.info(f"Found {len(attempted_accessions)} previously attempted datasets")
        
        # Get list of datasets that already have files
        existing_files_prefix = "data_files/"
        existing_blobs = list(bucket.list_blobs(prefix=existing_files_prefix))
        datasets_with_files = set()
        for blob in existing_blobs:
            parts = blob.name.split('/')
            if len(parts) >= 2:
                datasets_with_files.add(parts[1])
        
        logger.info(f"Found {len(datasets_with_files)} datasets with existing files")
        
        # Scan harvested datasets
        harvest_prefix = "harvest_results/"
        if source_filter:
            harvest_prefix = f"harvest_results/{source_filter}/"
        
        blobs = list(bucket.list_blobs(prefix=harvest_prefix))
        
        # Find datasets without files
        geo_datasets = []  # GEO datasets (higher priority - better file availability)
        other_datasets = []  # Other sources
        seen_accessions = set()
        
        for blob in blobs:
            if not blob.name.endswith('.json'):
                continue
            
            accession = blob.name.split('/')[-1].replace('.json', '')
            
            if accession in seen_accessions:
                continue
            seen_accessions.add(accession)
            
            # Skip if already has files or already attempted
            if accession in datasets_with_files:
                continue
            if accession in attempted_accessions:
                continue
            
            # Load metadata
            try:
                metadata = json.loads(blob.download_as_string())
                modality = metadata.get("modality", "bulk_rna")
                
                if modality_filter and modality != modality_filter:
                    continue
                
                ds_info = {
                    "accession": accession,
                    "modality": modality,
                    "source": blob.name.split('/')[1] if '/' in blob.name else "unknown",
                }
                
                # Categorize by source type
                if accession.startswith("GSE") or accession.startswith("GSM"):
                    geo_datasets.append(ds_info)
                elif accession.startswith("E-GEOD-"):
                    # E-GEOD are GEO mirrors - redirect to GEO format
                    geo_num = accession.replace("E-GEOD-", "")
                    ds_info["accession"] = f"GSE{geo_num}"
                    geo_datasets.append(ds_info)
                else:
                    other_datasets.append(ds_info)
                    
            except Exception as e:
                logger.debug(f"Could not load metadata for {accession}: {e}")
        
        # Combine datasets with GEO first if requested
        if geo_first:
            pending_datasets = geo_datasets + other_datasets
        else:
            pending_datasets = geo_datasets + other_datasets
            # Shuffle for variety
            import random
            random.shuffle(pending_datasets)
        
        # Limit to requested number
        pending_datasets = pending_datasets[:limit]
        
        logger.info(f"Found {len(pending_datasets)} datasets pending file download ({len(geo_datasets)} GEO, {len(other_datasets)} other)")
        
        if dry_run:
            # Preview mode
            discovery_results = []
            for ds in pending_datasets[:20]:  # Limit preview
                files = discover_dataset_files(ds["accession"])
                priority = filter_priority_files(files, ds["modality"], max_files)
                discovery_results.append({
                    "accession": ds["accession"],
                    "modality": ds["modality"],
                    "source": ds["source"],
                    "files_available": len(files),
                    "priority_files": len(priority),
                    "top_files": [
                        {"name": f.filename, "score": f.priority_score}
                        for f in priority[:3]
                    ],
                })
            
            return json.dumps({
                "status": "dry_run",
                "datasets_with_files": len(datasets_with_files),
                "datasets_pending": len(pending_datasets),
                "preview": discovery_results,
            }, indent=2), 200, {"Content-Type": "application/json"}
        
        # Download files
        results = []
        total_downloaded = 0
        processed_accessions = []
        
        for ds in pending_datasets:
            try:
                downloaded = download_dataset_files(
                    accession=ds["accession"],
                    modality=ds["modality"],
                    max_files=max_files,
                    max_size_mb=max_size_mb
                )
                
                count = sum(1 for f in downloaded if f.downloaded)
                total_downloaded += count
                
                results.append({
                    "accession": ds["accession"],
                    "modality": ds["modality"],
                    "files_downloaded": count,
                    "files": [f.gcs_path for f in downloaded if f.downloaded],
                })
                
                # Track as attempted (regardless of success)
                processed_accessions.append(ds["accession"])
                
            except Exception as e:
                logger.warning(f"Failed to download files for {ds['accession']}: {e}")
                results.append({
                    "accession": ds["accession"],
                    "error": str(e),
                })
                # Still mark as attempted to avoid retry loops
                processed_accessions.append(ds["accession"])
        
        # Save all processed accessions to tracking
        if processed_accessions:
            add_attempted_downloads(processed_accessions)
            logger.info(f"Marked {len(processed_accessions)} accessions as attempted")
        
        # Count successful vs no-files
        successful = sum(1 for r in results if r.get("files_downloaded", 0) > 0)
        no_files = sum(1 for r in results if r.get("files_downloaded", 0) == 0 and "error" not in r)
        errors = sum(1 for r in results if "error" in r)
        
        return json.dumps({
            "status": "success",
            "datasets_processed": len(results),
            "total_files_downloaded": total_downloaded,
            "summary": {
                "successful": successful,
                "no_files_available": no_files,
                "errors": errors,
            },
            "tracking": {
                "newly_attempted": len(processed_accessions),
                "total_attempted": len(attempted_accessions) + len(processed_accessions),
            },
            "results": results,
        }, indent=2), 200, {"Content-Type": "application/json"}
        
    except Exception as e:
        logger.error(f"Download pending files error: {e}")
        logger.debug(traceback.format_exc())
        return json.dumps({
            "status": "error",
            "error": str(e),
        }), 500, {"Content-Type": "application/json"}


# --- LOCAL DEVELOPMENT ENTRY POINT ---

if __name__ == "__main__":
    """Local development entry point."""
    import sys
    
    print("=" * 60)
    print("Multi-Omics Data Lake Harvester - Local Development Mode")
    print("=" * 60)
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "test":
            # Quick test with limited results
            print("\nRunning quick test harvest...")
            results = harvest_geo_modality(
                Modality.SCRNA,
                disease_context="trained immunity",
                max_results=5
            )
            print(f"\nFound {len(results)} results:")
            for r in results:
                print(f"  - {r.accession}: {r.title[:60]}...")
        
        elif command == "full":
            # Full harvest (be careful with rate limits)
            print("\nRunning full harvest (this may take a while)...")
            all_results = harvest_all_modalities(
                max_per_modality=20,
                enable_curation=False
            )
            for source, results in all_results.items():
                print(f"\n{source}: {len(results)} datasets")
        
        elif command == "modality":
            if len(sys.argv) > 2:
                modality_str = sys.argv[2]
                try:
                    modality = Modality(modality_str)
                    results = harvest_geo_modality(modality, max_results=10)
                    print(f"\n{modality.value}: {len(results)} datasets")
                    for r in results[:5]:
                        print(f"  - {r.accession}: {r.title[:60]}...")
                except ValueError:
                    print(f"Unknown modality: {modality_str}")
                    print(f"Available: {[m.value for m in Modality]}")
            else:
                print("Usage: python main.py modality <modality_name>")
        
        else:
            print(f"Unknown command: {command}")
            print("Available commands: test, full, modality <name>")
    else:
        print("\nUsage:")
        print("  python main.py test              - Quick test with 5 results")
        print("  python main.py full              - Full harvest (limited)")
        print("  python main.py modality <name>   - Harvest specific modality")
        print("\nAvailable modalities:")
        for m in Modality:
            print(f"  - {m.value}")

