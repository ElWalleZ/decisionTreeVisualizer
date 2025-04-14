import re
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
        """Limpia y escapa caracteres especiales para LaTeX"""
        # Escapa caracteres especiales
        content = content.replace("_", r"\_")
        content = content.replace("<=", r"$\leq$").replace(">", r"$>$")
        # Elimina espacios problemáticos
        content = re.sub(r"\s+", " ", content).strip()
        return content

    @staticmethod
    def parse_sklearn_tree(tree_text: str) -> TreeNode:
        lines = [line for line in tree_text.split('\n') if "|---" in line]
        stack = []
        root = None

        for line in lines:
            level = line.count('|   ')
            content = line.split('|--- ')[1].strip()

            # Formatea el contenido
            is_leaf = "class: " in content
            if is_leaf:
                content = f"Class: {content.split('class: ')[1]}"
            content = TreeConverter.clean_content(content)

            node = TreeNode(content=content, is_leaf=is_leaf)

            # Maneja la estructura del árbol
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
    def to_qtree_latex(root: TreeNode) -> str:
        """Genera código LaTeX usando qtree"""

        def build_qtree(node: TreeNode, depth: int = 0) -> str:
            if node is None:
                return ""

            indent = "  " * depth
            if node.is_leaf:
                return f"{indent}[.{node.content} ]\n"

            left = build_qtree(node.left, depth + 1)
            right = build_qtree(node.right, depth + 1)
            return f"{indent}[.{node.content}\n{left}{right}{indent}]\n"

        return r"""\documentclass[border=5pt]{standalone}
        \usepackage{qtree}
        \begin{document}
        \Tree
        """ + build_qtree(root) + r"""\end{document}"""

    @staticmethod
    def to_forest_latex(root: TreeNode) -> str:
        """Genera código LaTeX usando forest"""

        def build_forest(node: TreeNode) -> str:
            if node is None:
                return ""

            if node.is_leaf:
                return f"[\\textbf{{{node.content}}}]"

            left = build_forest(node.left)
            right = build_forest(node.right)
            return f"[{node.content} {left} {right}]"

        return r"""\documentclass[tikz,border=5pt]{standalone}
        \usepackage[edges]{forest}
        \usetikzlibrary{arrows.meta}
        
        \begin{document}
        \begin{forest}
        for tree={
            grow'=east,
            draw,
            edge path={
                \noexpand\path[\forestoption{edge}]
                (!u.parent anchor) -- +(5pt,0) |- (.child anchor)\forestoption{edge label};
            },
            if n children=0{
                fill=gray!20,
                rectangle,
                rounded corners=2pt
            }{},
            font=\small,
            s sep=10pt,
            l sep=15pt
        }
        """ + build_forest(root) + r"""
        \end{forest}
        \end{document}"""

    @staticmethod
    def convert(tree_text: str, package: str = 'forest') -> str:
        """Conversión completa con manejo de errores"""
        try:
            tree = TreeConverter.parse_sklearn_tree(tree_text)
            if package == 'forest':
                return TreeConverter.to_forest_latex(tree)
            return TreeConverter.to_qtree_latex(tree)
        except Exception as e:
            raise ValueError(f"Error converting tree: {str(e)}")