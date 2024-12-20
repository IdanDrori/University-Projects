# username - yuvalrubins
# id1      - 209281369
# name1    - Yuval Rubins
# id2      - 205983406
# name2    - Idan Drori


"""A class representing a node in an AVL tree"""


class AVLNode(object):
    """Constructor, you are allowed to add more fields.

    @type key: int or None
    @type value: any
    @param value: data of your node
    """

    def __init__(self, key, value):
        self.key = key
        self.value = value
        self.left = None
        self.right = None
        self.parent = None
        self.height = -1

    """returns the left child
    @rtype: AVLNode
    @returns: the left child of self, None if there is no left child (if self is virtual)
    """

    def get_left(self):
        return self.left

    """returns the right child

    @rtype: AVLNode
    @returns: the right child of self, None if there is no right child (if self is virtual)
    """

    def get_right(self):
        return self.right

    """returns the parent

    @rtype: AVLNode
    @returns: the parent of self, None if there is no parent
    """

    def get_parent(self):
        return self.parent

    """returns the key

    @rtype: int or None
    @returns: the key of self, None if the node is virtual
    """

    def get_key(self):
        return self.key

    """returns the value

    @rtype: any
    @returns: the value of self, None if the node is virtual
    """

    def get_value(self):
        return self.value

    """returns the height

    @rtype: int
    @returns: the height of self, -1 if the node is virtual
    """

    def get_height(self):
        return self.height

    """sets left child

    @type node: AVLNode
    @param node: a node
    """

    def set_left(self, node):
        self.left = node

    """sets right child

    @type node: AVLNode
    @param node: a node
    """

    def set_right(self, node):
        self.right = node

    """sets parent

    @type node: AVLNode
    @param node: a node
    """

    def set_parent(self, node):
        self.parent = node

    """sets key

    @type key: int or None
    @param key: key
    """

    def set_key(self, key):
        self.key = key

    """sets value

    @type value: any
    @param value: data
    """

    def set_value(self, value):
        self.value = value

    """sets the height of the node

    @type h: int
    @param h: the height
    """

    def set_height(self, h):
        self.height = h

    """
    update a node's height, computes via the childrens' heights
    """

    def update_height(self):
        self.set_height(max(self.get_right().get_height(),
                            self.get_left().get_height()) + 1)

    """returns whether self is not a virtual node
    @rtype: bool
    @returns: false if self is a virtual node, true otherwise.
    """

    def is_real_node(self):
        return self.get_key() is not None

    """returns the balance factor of the node
    @rtype: int
    @returns: balance factor
    """

    def get_balance_factor(self):
        return self.left.get_height() - self.right.get_height()

    """returns if the node is a left child of it's parent
    @rtype: bool
    @returns: True if node has a parent and it is it's left child, False otherwise
    """

    def is_left_child(self):
        return self.get_parent() is not None and self is self.get_parent().get_left()

    """returns if the node is a right child of it's parent
    @rtype: bool
    @returns: True if node has a parent and it is it's right child, False otherwise
    """

    def is_right_child(self):
        return self.get_parent() is not None and self is self.get_parent().get_right()


"""
A class implementing the ADT Dictionary, using an AVL tree.
"""


class AVLTree(object):
    """
    Constructor, you are allowed to add more fields.

    @type root: AVLNode
    """

    def __init__(self):
        self.root = None
        self._size = 0

    """
    Create a new tree with only a root in it and return it
    If root is None or not a real node, create an empty tree (without a root)

    @param root: node or root
    @type root: AVLNode
    @rtype: AVLTree
    @returns: the new created tree
    """

    @staticmethod
    def create_tree(root):
        tree = AVLTree()
        if root is not None and root.is_real_node():
            tree.root = root
            tree._size = 1
        return tree

    """
    Perform rebalancing operations (rotations or height update) for the given node up to the root
    to keep the tree to be an AVL tree
    Time Complexity: O(log(n))

    @param node: node to start rebalancing operations, and from it go upwards to the root
    @type root: AVLNode
    @param should_rebalance_only_once: True is only one rebalancing operation is required.
                                       Stop rebalancing after first rotation
    @type should_rebalance_only_once: bool
    @rtype: int
    @returns: number of rebalancing operations (rotations and height updates)
    """

    def rebalance_from_node(self, node: AVLNode, should_rebalance_only_once: bool):
        balance_count = 0
        while node is not None:
            bf = node.get_left().get_height() - node.get_right().get_height()
            next_node = node.get_parent()
            if abs(bf) < 2:
                if node.get_height() == max(node.get_left().get_height(), node.get_right().get_height()) + 1:
                    break
                else:
                    node.update_height()
                    balance_count += 1  # updating height increases rotate_count when not part of rotation
            else:
                balance_count += self.rotate(node, bf)
                if should_rebalance_only_once:
                    break

            node = next_node

        return balance_count

    """searches for a AVLNode in the dictionary corresponding to the key
    Time Complexity: O(log(n))

    @type key: int
    @param key: a key to be searched
    @rtype: AVLNode
    @returns: the AVLNode corresponding to key or None if key is not found.
    """

    def search(self, key):

        """recursive function to search for the node in the AVL tree
        Time Complexity: O(log(n))

        @type node: AVLNode
        @param node: the current node during the recursive search
        @type key: int
        @param key: the key to be searched
        @rtype: AVLNode
        @returns: the node corresponding to the key, or None if the key is not found
        """

        def search_rec(node, key):
            # key not found
            if node is None or node.get_key() is None:
                return None
            # key found
            if key == node.get_key():
                return node
            # key is in the left subtree
            if key < node.get_key():
                return search_rec(node.get_left(), key)
            # key is in the right subtree
            return search_rec(node.get_right(), key)

        return search_rec(self.root, key)

    """inserts val at position i in the dictionary
    Time Complexity: O(log(n))

    @type key: int
    @pre: key currently does not appear in the dictionary
    @param key: key of item that is to be inserted to self
    @type val: any
    @param val: the value of the item
    @rtype: int
    @returns: the number of rebalancing operation due to AVL rebalancing
    """

    def insert(self, key, val):
        self._size += 1

        # Initializing new node
        new_node = AVLNode(key, val)
        new_node.set_height(0)

        # adding the node in the standard BST way
        if self.root is None:
            self.root = new_node
        else:
            self.BST_insert(self.root, new_node)

        # Add two virtual children to the node (which was added as a leaf)
        new_node.set_right(AVLNode(None, None))
        new_node.right.set_parent(new_node)
        new_node.set_left(AVLNode(None, None))
        new_node.left.set_parent(new_node)

        return self.rebalance_from_node(new_node.get_parent(), True)

    """
    Inserts a new node into the AVL tree using standard BST rules
    Time Complexity: O(log(n))

    @type node: AVLNode
    @param node: current node during the insertion process
    @type newNode: AVLNode
    @param newNode: new node to be inserted
    """

    def BST_insert(self, node, new_node):
        if new_node.get_key() < node.get_key():
            if node.get_left().get_key() is None:
                node.set_left(new_node)
                new_node.set_parent(node)
            else:
                self.BST_insert(node.get_left(), new_node)
        else:
            if node.get_right().get_key() is None:
                node.set_right(new_node)
                new_node.set_parent(node)
            else:
                self.BST_insert(node.get_right(), new_node)

    """
    performs rotations to uphold the properties of an AVL tree

    @type node: AVLNode
    @param node: the node where the rotations are performed (it's position)
    @type bf: int
    @param bf: the node's balance factor
    @rtype: int
    @returns: returns the number of rotations done
    """

    def rotate(self, node, bf):
        # we reach this function only for nodes where |bf|==2
        if bf == 2:
            # what is the left son's bf? a lot of function calls, but I don't want to work directly with fields
            left_bf = node.get_left().get_balance_factor()
            if left_bf == -1:
                self.left_rotate(node.get_left())
                self.right_rotate(node)
                return 2
            else:
                self.right_rotate(node)
                return 1

        else:  # else bf == -2
            right_bf = node.get_right().get_balance_factor()
            if right_bf == 1:
                self.right_rotate(node.get_right())
                self.left_rotate(node)
                return 2
            else:
                self.left_rotate(node)
                return 1

    """
    performs a right rotation

    @type node: AVLNode
    @param node: the node around which we rotate
    """

    def right_rotate(self, node):
        tmp = node.left.right
        node.left.right = node
        node.left.parent = node.parent

        # check if the node was a left child or a right one to its parent
        if node.parent is not None:
            if node.parent.right.key == node.key:
                node.parent.right = node.left
            else:
                node.parent.left = node.left
        else:
            self.root = node.left
        node.parent = node.left
        node.left = tmp
        tmp.parent = node

        # all this pointer play is a bit confusing but this is just translating what we did in pseudo-code in class
        node.update_height()
        node.parent.update_height()

    """
    performs a left rotation

    @type node: AVLNode
    @param node: the node around which we rotate
    """

    def left_rotate(self, node):
        tmp = node.right.left
        node.right.left = node
        node.right.parent = node.parent

        # check if the node was a left child or a right one to its parent
        if node.parent is not None:
            if node.parent.right.key == node.key:
                node.parent.right = node.right
            else:
                node.parent.left = node.right
        else:
            self.root = node.right
        node.parent = node.right
        node.right = tmp
        tmp.parent = node

        # very similar confusing pointer play as right_rotate()
        node.update_height()
        node.parent.update_height()

    # note: there's a way to have both left_rotate and right_rotate in one function,
    #       but I don't know how necessary this is

    """
    deletes node from the dictionary
    Time Complexity: O(log(n))

    @type node: AVLNode
    @pre: node is a real pointer to a node in self
    @rtype: int
    @returns: the number of rebalancing operation due to AVL rebalancing
    """

    def delete(self, node):
        self._size -= 1

        # delete the node in the standard BST way
        parent = self.BST_delete(node)
        rotate_count = self.rebalance_from_node(parent, False)

        return rotate_count

    """deletes node in a standard BST way
    Time Complexity: O(log(n))

    @type node: AVLNode
    @param node: the node we want to delete
    """

    def BST_delete(self, node):
        # case 1: node is a leaf
        if not node.get_right().is_real_node() and not node.get_left().is_real_node():

            parent = node.get_parent()
            if parent is None:
                self.root = None
            else:
                virtual_node = AVLNode(None, None)
                virtual_node.set_parent(node.get_parent())
                if node.is_left_child():
                    # node to be deleted is left child
                    parent.set_left(virtual_node)
                else:
                    # node to be deleted is right child
                    parent.set_right(virtual_node)
                node.set_parent(None)

        # case 2: node has two children
        elif node.get_right().is_real_node() and node.get_left().is_real_node():
            successor = self.successor(node)
            parent = successor.get_parent()

            # remove successor from tree
            if successor.get_right().is_real_node():
                if successor.is_left_child():
                    parent.set_left(successor.get_right())
                else:
                    parent.set_right(successor.get_right())
                successor.get_right().set_parent(parent)
                deleted_node = successor.get_right()
            else:
                if successor.is_left_child():
                    parent.set_left(successor.get_left())
                else:
                    parent.set_right(successor.get_left())
                successor.get_left().set_parent(parent)
                deleted_node = successor.get_left()

            # replace node by successor
            successor.set_parent(node.get_parent())
            successor.set_right(node.get_right())
            successor.set_left(node.get_left())
            successor.set_height(node.get_height())

            # If the node is the root
            if node.get_parent() is None:
                self.root = successor
            else:
                if node.is_left_child():
                    # node to be deleted is left child
                    node.get_parent().set_left(successor)
                else:
                    # node to be deleted is right child
                    node.get_parent().set_right(successor)

            node.get_right().set_parent(successor)
            node.get_left().set_parent(successor)

            # Disconnect node from tree
            node.set_left(None)
            node.set_parent(None)
            node.set_right(None)
            parent = deleted_node.get_parent()

        # case 3: node has one child
        else:
            if node.get_right().is_real_node():
                child = node.get_right()
                node.set_right(None)
            else:  # left is real node
                child = node.get_left()
                node.set_left(None)

            parent = node.get_parent()
            if parent is None:
                # node to be deleted is root
                self.root = child
            else:
                if node.is_left_child():
                    # node to be deleted is left child
                    parent.set_left(child)
                else:
                    # node to be deleted is right child
                    parent.set_right(child)

            child.set_parent(parent)
            node.set_parent(None)

        return parent

    """
    finds successor of node
    Time Complexity: O(log(n))

    @type node: AVLNode
    @param node: the node who's successor we want to return
    @rtype: AVLNode
    @returns: the successor node
    """

    @staticmethod
    def successor(node):
        current = node
        if not current.get_right().is_real_node():
            # then going up until first right turn
            while current.get_parent() is not None and current.is_right_child():
                current = current.get_parent()
            return current.get_parent()
        else:
            # then node has a right child and we need to return the minimal node in the right subtree
            current = node.get_right()
            while current.is_real_node():
                current = current.get_left()

        return current.get_parent()

    """
    returns an array representing dictionary
    Time Complexity: O(n)

    @rtype: list
    @returns: a sorted list according to key of tuples (key, value) representing the data structure
    """

    def avl_to_array(self):

        """
        recursive function to traverse the AVL tree and populate an array with (key,value) tuples
        Time Complexity: O(n)

        @type node: AVLNode
        @param node: current node in AVL tree
        @type array: list
        @param array: the current tuple list of nodes
        @rtype: list
        @returns: tuple list of all nodes up to node (in-order traverse)
        """

        def avl_to_array_rec(node, array):
            # base case: tree is done
            if node is None or node.get_key() is None:
                return
            # left subtree
            avl_to_array_rec(node.get_left(), array)
            # inserting current node
            array.append((node.get_key(), node.get_value()))
            # right subtree
            avl_to_array_rec(node.get_right(), array)

        array = []
        avl_to_array_rec(self.get_root(), array)
        return array

    """
    returns the number of items in dictionary

    @rtype: int
    @returns: the number of items in dictionary
    """

    def size(self):
        return self._size

    """
    splits the dictionary at the i'th index
    Time Complexity: O( abs(Height(self)-Height(tree2) + 1))

    @type node: AVLNode
    @pre: node is in self
    @param node: The intended node in the dictionary according to whom we split
    @rtype: list
    @returns: a list [left, right], where left is an AVLTree representing the keys in the
    dictionary smaller than node.key, right is an AVLTree representing the keys in the
    dictionary larger than node.key.
    """

    def split(self, node):
        left = self.create_tree(node.get_left())
        right = self.create_tree(node.get_right())
        left_subtrees = []
        right_subtrees = []
        left_subroots = []
        right_subroots = []
        current = node

        # Disconnect node from it's children
        node.get_left().set_parent(None)
        node.get_right().set_parent(None)
        node.set_left(None)
        node.set_right(None)

        # Go up the tree and and divide it to left and right subtrees
        while current.get_parent() is not None:
            parent = current.get_parent()
            if current.is_left_child():
                # Parent is greater than child, therefore on the right side
                right_subroots.append(parent)
                right_subtrees.append(self.create_tree(parent.get_right()))
            else:
                # Parent is smaller than child, therefore on the left side
                left_subroots.append(parent)
                left_subtrees.append(self.create_tree(parent.get_left()))

            # Disconnect all connections between parent and it's children
            parent.get_left().set_parent(None)
            parent.get_right().set_parent(None)
            parent.set_left(None)
            parent.set_right(None)
            current = parent

        # Join all left trees
        for i, subroot in enumerate(left_subroots):
            left.join_node(left_subtrees[i], subroot)

        # Join all right trees
        for i, subroot in enumerate(right_subroots):
            right.join_node(right_subtrees[i], subroot)

        return left, right

    """
    joins self with key and another AVLTree
    Time Complexity: O( abs(Height(self)-Height(tree2) + 1))

    @type tree2: AVLTree
    @param tree2: a dictionary to be joined with self
    @type key: int
    @param key: The key separating self with tree2
    @type val: any
    @param val: The value attached to key
    @pre: all keys in self are smaller than key and all keys in tree2 are larger than key
    @rtype: int
    @returns: the absolute value of the difference between the height of the AVL trees joined
    """

    def join(self, tree2, key, val):
        node = AVLNode(key, val)
        return self.join_node(tree2, node)

    """
    joins self with new node (not in any tree) and another AVLTree
    Time Complexity: O( abs(Height(self)-Height(tree2) + 1))

    @type tree2: AVLTree
    @param tree2: a dictionary to be joined with self
    @type node: AVLNode
    @param node: new node in the middle to join between the trees
    @pre: node is not virtual or None
    @rtype: int
    @returns: the absolute value of the difference between the height of the AVL trees joined
    """

    def join_node(self, tree2, node):
        new_subtree_root = node

        # If tree is empty, set it to be a virtual node
        if self.get_root() is None:
            self.root = AVLNode(None, None)
        if tree2.get_root() is None:
            tree2.root = AVLNode(None, None)

        # Update size of tree
        self._size = self.size() + tree2.size() + 1
        height_tree1 = self.get_root().get_height()
        height_tree2 = tree2.get_root().get_height()

        # If trees are same height, just make the new node their root and add them from each side
        if height_tree1 == height_tree2:
            self.get_root().set_parent(new_subtree_root)
            tree2.get_root().set_parent(new_subtree_root)
            if self.get_root().is_real_node() and self.get_root().get_key() < node.get_key():
                # if self < key < tree2
                new_subtree_root.set_left(self.get_root())
                new_subtree_root.set_right(tree2.get_root())
            else:
                # if self > key > tree2
                new_subtree_root.set_right(self.get_root())
                new_subtree_root.set_left(tree2.get_root())
            self.root = new_subtree_root
            self.get_root().update_height()
            return abs(height_tree1 - height_tree2) + 1

        # Always tree2 will be the shorter tree that we join to self which will be the taller
        if height_tree1 < height_tree2:
            self.root, tree2.root = tree2.get_root(), self.get_root()

        is_new_tree_smaller = node.get_key() < self.root.get_key()
        current_node = self.get_root()

        # Descend on the left or right branch of the big tree
        # until the height is almost like the small tree's height
        while current_node.get_height() > tree2.get_root().get_height() and current_node.is_real_node():
            if is_new_tree_smaller:  # tree2 < self
                current_node = current_node.get_left()
            else:  # tree2 > self
                current_node = current_node.get_right()

        # Add the new node as parent of both found node in big tree and root of small tree
        # and connect it to the main tree
        parent_node = current_node.get_parent()
        new_subtree_root.set_parent(parent_node)
        tree2.get_root().set_parent(new_subtree_root)
        current_node.set_parent(new_subtree_root)

        # Add the two subtrees as children of new node
        if is_new_tree_smaller:  # tree2 < self
            new_subtree_root.set_left(tree2.root)
            new_subtree_root.set_right(current_node)
            parent_node.set_left(new_subtree_root)
        else:  # tree2 > self
            new_subtree_root.set_right(tree2.root)
            new_subtree_root.set_left(current_node)
            parent_node.set_right(new_subtree_root)

        new_subtree_root.update_height()
        self.rebalance_from_node(new_subtree_root.get_parent(), False)
        return abs(height_tree1 - height_tree2) + 1

    """
    returns the root of the tree representing the dictionary

    @rtype: AVLNode
    @returns: the root, None if the dictionary is empty
    """

    def get_root(self):
        return self.root
