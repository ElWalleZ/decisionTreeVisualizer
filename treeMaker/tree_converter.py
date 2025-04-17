import re, os
import subprocess
from dataclasses import dataclass
from typing import Optional


@dataclass
class TreeNode:
    """
    Representa un nodo en un árbol binario.
    Atributos:
    content (str): Contenido del nodo.
    left (Optional[TreeNode]): Hijo izquierdo.
    right (Optional[TreeNode]): Hijo derecho.
    is_leaf (bool): Indica si el nodo es una hoja.
    """
    content: str
    left: Optional['TreeNode'] = None
    right: Optional['TreeNode'] = None
    is_leaf: bool = False


class TreeConverter:
    @staticmethod
    def clean_content(content: str) -> str:
        """
        Limpia y formatea el contenido para LaTeX.
        Args:
        content (str): Texto a limpiar.
        Returns:
        str: Contenido formateado para LaTeX.
        """
        content = content.replace("_", r"\_")
        content = content.replace("<=", r" $\leq$ ").replace(">", r" $>$ ")
        content = re.sub(r"\s+", " ", content).strip()

        if "class: " in content.lower():
            content = r"\textbf{" + content.replace("class: ", "").strip() + "}"
        return content

    @staticmethod
    def parse_sklearn_tree(tree_text: str) -> TreeNode:
        """
        Parsea texto de árbol de scikit-learn a estructura TreeNode.
        Args:
        tree_text (str): Texto del árbol de scikit-learn.
        Returns:
        TreeNode: Raíz del árbol parseado.
        """
        lines = [line for line in tree_text.split('\n') if "|---" in line]
        stack = []
        root = None

        for line in lines:
            clean_line = re.sub(r'^[|\s]+', '', line)
            content = re.sub(r'^---\s*', '', clean_line).strip()
            is_leaf = "class: " in content.lower()

            level = len(re.findall(r'\|   ', line))

            node = TreeNode(
                content=content,
                is_leaf=is_leaf
            )

            while len(stack) > level:
                stack.pop()

            if not stack:
                if root is None:
                    root = node
                else:
                    root.right = node
            else:
                parent = stack[-1]
                if parent.left is None:
                    parent.left = node
                else:
                    parent.right = node

            stack.append(node)

        def clean_node(node):
            """Manda a limpiar el contenido de los nodos"""
            if node:
                node.content = TreeConverter.clean_content(node.content)
                clean_node(node.left)
                clean_node(node.right)

        clean_node(root)
        return root

    @staticmethod
    def to_vertical_forest(tree: TreeNode) -> str:
        """
        Genera código LaTeX para visualizar el árbol con el paquete forest.
        Args:
        tree (TreeNode): Raíz del árbol a convertir.
        Returns:
        str: Código LaTeX completo para el árbol.
        """
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
        """
        Convierte texto de árbol scikit-learn a PDF.
        Args:
        tree_text (str): Texto del árbol.
        output_dir (str): Directorio de salida.
        filename (str): Nombre base del archivo.
        Returns:
        str: Ruta al archivo PDF generado.
        """
        try:

            print("\n=== TEXTO ORIGINAL DEL ÁRBOL ===")
            print(tree_text)

            tree = TreeConverter.parse_sklearn_tree(tree_text)
            latex_code = TreeConverter.to_vertical_forest(tree)

            print("\n=== ESTRUCTURA DEL ÁRBOL ANALIZADA ===")
            TreeConverter.print_tree_debug(tree)

            os.makedirs(output_dir, exist_ok=True)
            tex_path = os.path.join(output_dir, f"{filename}.tex")

            with open(tex_path, 'w', encoding='utf-8') as f:
                f.write(latex_code)

            # Compilar
            for _ in range(2):
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

    @staticmethod
    def print_tree_debug(tree: TreeNode, level: int = 0):
        """
        Muestra el árbol en consola con formato legible.
        Args:
        tree (TreeNode): Raíz del árbol a mostrar.
        level (int): Nivel de indentación.
        """
        indent = "    " * level
        if tree.is_leaf:
            print(f"{indent}└── [Hoja]: {tree.content.replace('\\textbf{', '').replace('}', '')}")
        else:
            print(f"{indent}├── [Decisión]: {tree.content}")
            if tree.left:
                TreeConverter.print_tree_debug(tree.left, level + 1)
            if tree.right:
                TreeConverter.print_tree_debug(tree.right, level + 1)