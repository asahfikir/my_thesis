import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# Set up the figure with high DPI for better quality
fig, ax = plt.subplots(figsize=(16, 12), dpi=150)
ax.set_xlim(0, 16)
ax.set_ylim(0, 12)
ax.axis('off')

# Color scheme - professional academic colors
color_input = '#E8F4FD'
color_process = '#FFF4E6'
color_architecture = '#F0F9E8'
color_output = '#E8F5E9'
color_evaluation = '#F3E5F5'
color_arrow = '#37474F'
color_text = '#263238'
color_variable = '#BF360C'

# Title
ax.text(8, 11.5, 'KERANGKA PEMIKIRAN', fontsize=20, weight='bold', 
        ha='center', va='center', color='#1A237E')
ax.text(8, 11.1, 'Sistem RAG dengan Small Language Models untuk Dokumen Bahasa Indonesia', 
        fontsize=13, ha='center', va='center', color='#455A64', style='italic')

# 1. INPUT BOX
input_box = FancyBboxPatch((0.5, 9.5), 3, 1.2, 
                          boxstyle="round,pad=0.1", 
                          facecolor=color_input, edgecolor='#1565C0', linewidth=2)
ax.add_patch(input_box)
ax.text(2, 10.1, 'INPUT', fontsize=12, weight='bold', ha='center', color='#1565C0')
ax.text(2, 9.85, 'Dokumen Bahasa Indonesia', fontsize=10, ha='center', color=color_text)
ax.text(2, 9.65, '(PDF/TXT)', fontsize=9, ha='center', color='#607D8B')

# 2. PREPROCESSING BOX (Process 1)
process1_box = FancyBboxPatch((0.5, 7), 3, 2, 
                             boxstyle="round,pad=0.1", 
                             facecolor=color_process, edgecolor='#E65100', linewidth=2)
ax.add_patch(process1_box)
ax.text(2, 8.7, 'PROSES 1', fontsize=11, weight='bold', ha='center', color='#E65100')
ax.text(2, 8.4, 'PREPROCESSING', fontsize=11, weight='bold', ha='center', color='#E65100')
ax.text(2, 7.9, 'Eksplisitasi Variabel:', fontsize=10, weight='bold', ha='center', color=color_text)

# Variables inside Process 1
var_a = FancyBboxPatch((0.8, 7.4), 2.4, 0.4, boxstyle="round,pad=0.05", 
                       facecolor='#FFCCBC', edgecolor=color_variable, linewidth=1.5)
ax.add_patch(var_a)
ax.text(2, 7.6, 'A. Chunk Size (VI₁)', fontsize=9, ha='center', color=color_variable, weight='bold')

var_b = FancyBboxPatch((0.8, 6.9), 2.4, 0.4, boxstyle="round,pad=0.05", 
                       facecolor='#FFCCBC', edgecolor=color_variable, linewidth=1.5)
ax.add_patch(var_b)
ax.text(2, 7.1, 'B. Overlap Size (VI₂)', fontsize=9, ha='center', color=color_variable, weight='bold')

# 3. VECTORIZATION BOX (Process 2)
process2_box = FancyBboxPatch((0.5, 4.5), 3, 1.8, 
                             boxstyle="round,pad=0.1", 
                             facecolor=color_process, edgecolor='#E65100', linewidth=2)
ax.add_patch(process2_box)
ax.text(2, 5.9, 'PROSES 2', fontsize=11, weight='bold', ha='center', color='#E65100')
ax.text(2, 5.6, 'VECTORIZATION', fontsize=11, weight='bold', ha='center', color='#E65100')
ax.text(2, 5.1, 'Konversi teks menjadi', fontsize=9, ha='center', color=color_text)
ax.text(2, 4.8, 'Embedding Vector', fontsize=9, ha='center', color=color_text)
ax.text(2, 4.5, '(Model Bahasa Indonesia)', fontsize=8, ha='center', color='#607D8B', style='italic')

# 4. RAG ARCHITECTURE (Main Component - Center)
rag_main = FancyBboxPatch((5.5, 3.5), 5, 5, 
                         boxstyle="round,pad=0.1", 
                         facecolor=color_architecture, edgecolor='#2E7D32', linewidth=3)
ax.add_patch(rag_main)
ax.text(8, 8.1, 'ARSITEKTUR RAG', fontsize=14, weight='bold', ha='center', color='#1B5E20')

# Sub-components within RAG
# User Query
query_box = FancyBboxPatch((6, 7), 4, 0.8, boxstyle="round,pad=0.05", 
                          facecolor='#C8E6C9', edgecolor='#2E7D32', linewidth=1.5)
ax.add_patch(query_box)
ax.text(8, 7.4, 'Query User', fontsize=10, weight='bold', ha='center', color='#2E7D32')

# Vector Database
vector_db = FancyBboxPatch((6, 5.6), 4, 0.9, boxstyle="round,pad=0.05", 
                          facecolor='#A5D6A7', edgecolor='#2E7D32', linewidth=1.5)
ax.add_patch(vector_db)
ax.text(8, 6.3, 'Vector Database', fontsize=10, weight='bold', ha='center', color='#2E7D32')
ax.text(8, 6.0, '(Retrieval & Similarity Search)', fontsize=8, ha='center', color='#1B5E20')

# Context Injection
context_box = FancyBboxPatch((6, 4.4), 4, 0.9, boxstyle="round,pad=0.05", 
                            facecolor='#81C784', edgecolor='#2E7D32', linewidth=1.5)
ax.add_patch(context_box)
ax.text(8, 4.95, 'Context Injection', fontsize=10, weight='bold', ha='center', color='#2E7D32')
ax.text(8, 4.7, 'Prompt Engineering', fontsize=8, ha='center', color='#1B5E20')

# SLM Generation
slm_box = FancyBboxPatch((6, 3.2), 4, 1.0, boxstyle="round,pad=0.05", 
                        facecolor='#66BB6A', edgecolor='#1B5E20', linewidth=2)
ax.add_patch(slm_box)
ax.text(8, 3.9, 'Model Generasi (SLM)', fontsize=10, weight='bold', ha='center', color='#1B5E20')
ax.text(8, 3.65, 'Phi-2 / Gemma-2', fontsize=9, ha='center', color='#2E7D32')
ax.text(8, 3.4, 'Lokal (Laptop Yoga 7i)', fontsize=8, ha='center', color='#2E7D32', style='italic')

# 5. OUTPUT BOX
output_box = FancyBboxPatch((12, 5.5), 3.5, 1.2, 
                           boxstyle="round,pad=0.1", 
                           facecolor=color_output, edgecolor='#2E7D32', linewidth=2)
ax.add_patch(output_box)
ax.text(13.75, 6.3, 'OUTPUT', fontsize=12, weight='bold', ha='center', color='#2E7D32')
ax.text(13.75, 5.9, 'Jawaban yang Dihasilkan', fontsize=10, ha='center', color=color_text)

# 6. EVALUATION BOX (Dependent Variables)
eval_box = FancyBboxPatch((5.5, 0.5), 5, 2.5, 
                         boxstyle="round,pad=0.1", 
                         facecolor=color_evaluation, edgecolor='#7B1FA2', linewidth=2)
ax.add_patch(eval_box)
ax.text(8, 2.7, 'EVALUASI', fontsize=12, weight='bold', ha='center', color='#7B1FA2')
ax.text(8, 2.4, '(Variabel Dependen)', fontsize=10, ha='center', color='#7B1FA2', style='italic')
ax.text(8, 2.0, 'Metrik RAGAS:', fontsize=10, weight='bold', ha='center', color=color_text)

# RAGAS Metrics
faith_box = FancyBboxPatch((5.8, 1.5), 2.1, 0.4, boxstyle="round,pad=0.05", 
                          facecolor='#E1BEE7', edgecolor='#7B1FA2', linewidth=1.5)
ax.add_patch(faith_box)
ax.text(6.85, 1.7, '• Faithfulness', fontsize=9, ha='center', color='#4A148C')

rel_box = FancyBboxPatch((5.8, 1.0), 2.1, 0.4, boxstyle="round,pad=0.05", 
                        facecolor='#E1BEE7', edgecolor='#7B1FA2', linewidth=1.5)
ax.add_patch(rel_box)
ax.text(6.85, 1.2, '• Answer Relevancy', fontsize=9, ha='center', color='#4A148C')

prec_box = FancyBboxPatch((5.8, 0.5), 2.1, 0.4, boxstyle="round,pad=0.05", 
                         facecolor='#F3E5F5', edgecolor='#9C27B0', linewidth=1.5, linestyle='--')
ax.add_patch(prec_box)
ax.text(6.85, 0.7, '• Context Precision (Opsional)', fontsize=8, ha='center', color='#7B1FA2')

# Legend linking variables
legend_text = FancyBboxPatch((8.1, 1.5), 2.3, 0.9, boxstyle="round,pad=0.05", 
                            facecolor='white', edgecolor='#757575', linewidth=1)
ax.add_patch(legend_text)
ax.text(9.25, 2.2, 'VI = Variabel Independen', fontsize=8, ha='center', color='#555')
ax.text(9.25, 1.9, 'VD = Variabel Dependen', fontsize=8, ha='center', color='#555')

# Arrows between components
arrow_style = "Simple, tail_width=0.5, head_width=4, head_length=8"

# Input -> Process 1
ax.annotate('', xy=(2, 9.5), xytext=(2, 10.1),
            arrowprops=dict(arrowstyle='->', color=color_arrow, lw=2))

# Process 1 -> Process 2
ax.annotate('', xy=(2, 7), xytext=(2, 7.6),
            arrowprops=dict(arrowstyle='->', color=color_arrow, lw=2))

# Process 2 to RAG
ax.annotate('', xy=(5.5, 5.4), xytext=(3.5, 5.4),
            arrowprops=dict(arrowstyle='->', color=color_arrow, lw=2))
ax.text(4.5, 5.6, 'Embedding', fontsize=8, ha='center', color='#555')

# RAG internal flow arrows
ax.annotate('', xy=(8, 7), xytext=(8, 6.6),
            arrowprops=dict(arrowstyle='->', color='#2E7D32', lw=1.5))
ax.annotate('', xy=(8, 5.6), xytext=(8, 5.3),
            arrowprops=dict(arrowstyle='->', color='#2E7D32', lw=1.5))
ax.annotate('', xy=(8, 4.4), xytext=(8, 4.1),
            arrowprops=dict(arrowstyle='->', color='#2E7D32', lw=1.5))

# RAG -> Output
ax.annotate('', xy=(12, 6.1), xytext=(10.5, 6.1),
            arrowprops=dict(arrowstyle='->', color=color_arrow, lw=2))

# Output -> Evaluation (feedback loop)
ax.annotate('', xy=(13.75, 5.5), xytext=(13.75, 3),
            arrowprops=dict(arrowstyle='->', color='#7B1FA2', lw=1.5, linestyle='--'))
ax.text(14.5, 4.3, 'Evaluasi', fontsize=9, ha='center', color='#7B1FA2', rotation=90)

# Process 2 -> Evaluation (dotted line showing testing)
ax.plot([2, 2, 5.5], [4.5, 1.75, 1.75], 'k--', alpha=0.3, lw=1)
ax.annotate('', xy=(5.5, 1.75), xytext=(3, 1.75),
            arrowprops=dict(arrowstyle='->', color='#999', lw=1, linestyle='--'))

# Add variable arrows from Process 1 to RAG (showing influence)
ax.annotate('', xy=(5.5, 8), xytext=(3.5, 8),
            arrowprops=dict(arrowstyle='->', color=color_variable, lw=1.5, linestyle=':'))
ax.text(4.5, 8.2, 'Chunk & Overlap', fontsize=8, ha='center', color=color_variable)

# Box labels
ax.text(0.3, 10.1, '1', fontsize=14, weight='bold', color='#1565C0')
ax.text(0.3, 8, '2', fontsize=14, weight='bold', color='#E65100')
ax.text(0.3, 5.4, '3', fontsize=14, weight='bold', color='#E65100')
ax.text(5.3, 8.2, '4', fontsize=14, weight='bold', color='#1B5E20')
ax.text(12.3, 6.1, '5', fontsize=14, weight='bold', color='#2E7D32')
ax.text(5.3, 2.5, '6', fontsize=14, weight='bold', color='#7B1FA2')

plt.tight_layout()
plt.savefig('/mnt/kimi/output/kerangka_pemikiran_rag.png', bbox_inches='tight', dpi=300, facecolor='white')
plt.show()
