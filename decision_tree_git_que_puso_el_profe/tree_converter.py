import re, os
import subprocess
from dataclasses import dataclass
from typing import Optional


@dataclass
class TreeNode:
    content: str
    left: Optional['TreeNode'] = None
    right: Optional['TreeNode'] = None
    is_leaf: bool = False


class TreeConverter:
    @staticmethod
    def clean_content(content: str) -> str:
        """Limpia y formatea el contenido para LaTeX"""
        content = content.replace("_", r"\_")
        content = content.replace("<=", r" $\leq$ ").replace(">", r" $>$ ")
        content = re.sub(r"\s+", " ", content).strip()

        # Formato especial para nodos hoja
        if "class: " in content.lower():
            content = r"\textbf{" + content.replace("class: ", "").strip() + "}"
        return content

    @staticmethod
    def parse_sklearn_tree(tree_text: str) -> TreeNode:
        lines = [line for line in tree_text.split('\n') if "|---" in line]
        stack = []
        root = None

        for line in lines:
            level = line.count('|   ')
            content = line.split('|--- ')[1].strip()
            is_leaf = "class: " in content.lower()

            node = TreeNode(
                content=TreeConverter.clean_content(content),
                is_leaf=is_leaf
            )

            # Manejar estructura del árbol
            while len(stack) > level:
                stack.pop()

            if stack:
                parent = stack[-1]
                if parent.left is None:
                    parent.left = node
                else:
                    parent.right = node
            else:
                root = node

            stack.append(node)

        return root

    @staticmethod
    def to_vertical_forest(tree: TreeNode) -> str:
        """Genera árbol vertical con mejor espaciado"""

        def build_branches(node: TreeNode) -> str:
            if node is None:
                return ""

            if node.is_leaf:
                return f"[\\textbf{{{node.content}}}]"

            left = build_branches(node.left)
            right = build_branches(node.right)

            return f"[{node.content} {left} {right}]"

        return r"""\documentclass[tikz,border=10pt]{standalone}
    \usepackage[edges]{forest}
    \usetikzlibrary{arrows.meta}

    \begin{document}
    \begin{forest}
    for tree={
        grow=south,
        parent anchor=south,
        child anchor=north,
        draw,
        edge={->,>=latex},
        if n children=0{
            fill=green!10,
            rounded corners=3pt,
            font=\small\bfseries
        }{
            fill=blue!5,
            rounded corners=2pt,
            font=\small
        },
        edge path={
            \noexpand\path[\forestoption{edge}]
            (!u.parent anchor) -- +(0,-8pt) -| (.child anchor)\forestoption{edge label};
        },
        l sep=20pt,
        s sep=15pt,
        tier/.wrap pgfmath arg={tier #1}{level()},
        where level=0{
            font=\large\bfseries
        }{}
    }
    """ + build_branches(tree) + r"""
    \end{forest}
    \end{document}"""

    @staticmethod
    def convert_to_pdf(tree_text: str, output_dir: str, filename: str) -> str:
        """Conversión completa a PDF con manejo de errores"""
        try:
            tree = TreeConverter.parse_sklearn_tree(tree_text)
            latex_code = TreeConverter.to_vertical_forest(tree)

            # Guardar archivo .tex
            os.makedirs(output_dir, exist_ok=True)
            tex_path = os.path.join(output_dir, f"{filename}.tex")

            with open(tex_path, 'w', encoding='utf-8') as f:
                f.write(latex_code)

            # Compilar
            for _ in range(2):  # Compilar 2 veces para referencias
                result = subprocess.run(
                    ["pdflatex", "-interaction=nonstopmode", tex_path],
                    cwd=output_dir,
                    capture_output=True,
                    text=True
                )
                if result.returncode != 0:
                    raise RuntimeError(result.stderr)

            return os.path.join(output_dir, f"{filename}.pdf")

        except Exception as e:
            raise RuntimeError(f"Error generating tree: {str(e)}")