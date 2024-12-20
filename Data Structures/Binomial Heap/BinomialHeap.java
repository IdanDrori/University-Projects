/**
 * BinomialHeap
 *
 * An implementation of binomial heap over non-negative integers.
 * Based on exercise from previous semester.
 */
public class BinomialHeap
{
	public int size;
	public HeapNode last;
	public HeapNode min;
	private static final int MIN_VALUE = -1;

	// Default Constructor
	public BinomialHeap() {
		this.size = 0;
		this.last = null;
		this.min = null;
	}

	public BinomialHeap(int size, HeapNode last, HeapNode min) { // Constructor
		this.size = size;
		this.last = last;
		this.min = min;
	}

	private int rankToSize(int rank) {
		return (int)Math.pow(2, rank);
	}

	public HeapNode getLast() {
		return this.last;
	}

	public HeapNode getMin() {
		return this.min;
	}

	public void setSize(int size) {
		this.size = size;
	}

	public void setLast(HeapNode last) {
		this.last = last;
	}

	public void setMin(HeapNode min) {
		this.min = min;
	}

	/**
	 *
	 * pre: key > 0
	 *
	 * Insert (key,info) into the heap and return the newly generated HeapItem.
	 *
	 */
	public HeapItem insert(int key, String info)
	{
		HeapItem new_item = new HeapItem(key, info);
		HeapNode new_node = new HeapNode(new_item);
		new_item.setNode(new_node);
		new_node.setNext(new_node);

		// If heap is empty just set the new node to it
		if (this.empty()) {
			this.setLast(new_node);
			this.setMin(new_node);
			this.setSize(1);
			return new_item;
		}

		BinomialHeap new_heap = new BinomialHeap(1, new_node, new_node);

		int expected_rank = 0;
		HeapNode prev_node = this.getLast();
		HeapNode current_node = this.getLast().getNext();

		// Find the longest sequence of trees with incrementing ranks, starting from 0
		while (current_node.getRank() == expected_rank) {
			prev_node = current_node;
			current_node = current_node.getNext();
			expected_rank++;
		}

		if (prev_node == this.getLast()) { // If all the trees are in the sequence meld all the heap
			this.meld(new_heap);
		}
		else {
			// Create a subheap only from trees in the sequence and meld only it

			// Disconnect subheap from rest of heap
			prev_node.setNext(this.getLast().getNext());
			BinomialHeap subheap = new BinomialHeap(rankToSize(expected_rank) - 1, prev_node, null);
			subheap.meld(new_heap);

			// Connect back the subheap to rest of heap
			this.getLast().setNext(subheap.getLast().getNext());
			subheap.getLast().setNext(current_node);

			// Update heap parameters
			this.setSize(this.size() + 1);
			if ((subheap.findMin().getKey() == this.findMin().getKey() && this.getMin().getParent() != null) ||
				(subheap.findMin().getKey() < this.findMin().getKey())) {
				this.setMin(subheap.getMin());
			}
		}

		return new_item;
	}

	/**
	 *
	 * Delete the minimal item
	 * Complexity: O(log n)
	 */
	public void deleteMin()
	{
		if (this.empty()) {
			return;
		}

		HeapNode originalMin = this.getMin();
		HeapNode beforeMin = this.getMin().getNext();

		// disconnect min's tree from heap
		while (beforeMin.getNext() != originalMin) {
			beforeMin = beforeMin.getNext();
		}

		beforeMin.setNext(originalMin.getNext());
		originalMin.setNext(null);

		if (originalMin == beforeMin) { // Only one tree in the heap which we erase it's root
			this.setMin(null);
			this.setLast(null);
			this.setSize(0);
		}
		else {
			// update original heap (min, last, size)
			if (this.getLast() == originalMin) {
				this.setLast(beforeMin);
			}
			this.updateMin();
			this.setSize(this.size() - this.rankToSize(originalMin.getRank()));
		}

		// create new heap from min's children
		HeapNode currentChild = originalMin.getChild();
		if (currentChild != null) {
			do  {
				currentChild.setParent(null);
				currentChild = currentChild.getNext();
			} while (currentChild != originalMin.getChild());
		}

		BinomialHeap newHeap = new BinomialHeap(this.rankToSize(originalMin.getRank()) - 1,
											    originalMin.getChild(),
												null);
		newHeap.updateMin();
		originalMin.setChild(null);

		// meld the new heap to the original one
		this.meld(newHeap);
	}

	/**
	 *
	 * Return the minimal HeapItem
	 *
	 */
	public HeapItem findMin()
	{
		if (this.empty()) {
			return null;
		}
		else {
			return this.getMin().getItem();
		}
	}

	/**
	 *
	 * pre: 0 < diff < item.key
	 *
	 * Decrease the key of item by diff and fix the heap.
	 *
	 */
	public void decreaseKey(HeapItem item, int diff)
	{
		item.setKey(item.getKey() - diff); // decreasing key
		HeapNode cur_node = item.getNode();

		// while current node still has parent we sift up
		// if the parent's key is smaller then we're done
		while (cur_node.getParent() != null &&
				cur_node.getItem().getKey() < cur_node.getParent().getItem().getKey()) {
			cur_node.swap(cur_node.getParent()); // swaps current node and its parent
			cur_node = cur_node.getParent();
		}

		if (cur_node.getItem().getKey() < this.getMin().getItem().getKey()) {
			this.setMin(cur_node); // if current node has a smaller key than the minimum then set it as the new minimum
		}
	}

	/**
	 *
	 * Delete the item from the heap.
	 * Complexity: O(log n)
	 */
	public void delete(HeapItem item)
	{
		int diff = item.getKey() - MIN_VALUE;
		this.decreaseKey(item, diff);
		this.deleteMin();
	}

	/**
	 *
	 * Meld the heap with heap2
	 *
	 */
	public void meld(BinomialHeap heap2)
	{
		if (heap2.empty()) { // If second heap is empty (or both heaps are empty) then there's nothing to do
			return;
		}
		if (this.empty() && !heap2.empty()) { // Melding empty heap with non-empty heap.
			this.setSize(heap2.size());
			this.setLast(heap2.getLast());
			this.setMin(heap2.getMin());
			return;
		}

		// Meld two non-empty heaps
		this.setSize(this.size() + heap2.size()); // Updating size for new heap

		// length of tree list will be the maximum of the number of trees of the heaps + 1 to account for a possible carry later on
		int highest_rank = this.highest_rank();
		int len = highest_rank + 1;
		HeapNode[] this_trees = this.getTrees(len);
		HeapNode[] heap2_trees = heap2.getTrees(len);
		HeapNode carry = null; // initializing carry

		for (int i = 0; i < len; i++) { // iterating over ranks
			if (carry == null) { // no carry
				if (heap2_trees[i] == null) { // if heap2 doesn't have a tree at this rank we continue in the loop
					continue;
				}
				if (this_trees[i] == null) { // if only heap2 has a tree in rank i
					this_trees[i] = heap2_trees[i];
				}
				else { // both heaps have tree in rank i
					carry = link(this_trees[i], heap2_trees[i]);
					this_trees[i] = null;
				}
			}
			else { // there's a carry
				if (this_trees[i] == null && heap2_trees[i] == null) {
					this_trees[i] = carry;
					carry = null;
				}

				else if (heap2_trees[i] == null) {
					carry = link(this_trees[i], carry);
					this_trees[i] = null;
				}
				else if (this_trees[i] == null) {
					carry = link(heap2_trees[i], carry);
					this_trees[i] = null;
				}
				else { // there's a carry and both heap2 and the current heap have a tree at this rank
					carry = link(heap2_trees[i], carry); // keeping the tree from the main heap in the current rank, and the new carry will be of rank+1
				}
			}
		}

		this.setLast(this_trees[highest_rank]);

		// link all trees together
		HeapNode current_node = this.getLast();
		for (int i = 0; i < len; i++) {
			if (this_trees[i] != null) {
				current_node.setNext(this_trees[i]);
				current_node = this_trees[i];
			}
		}

		this.updateMin();
	}

	/**
	 * Returns list of given length trees in the heap
	 * The i-th element of the list is the tree at rank i
	 * Time Complexity: O(log n)
	 * */
	public HeapNode[] getTrees(int len) {
		HeapNode[] tree_list = new HeapNode[len];
		HeapNode node = this.getLast();
		tree_list[node.getRank()] = node;

		node = node.getNext();
		while (node != this.getLast()) {
			tree_list[node.getRank()] = node;
			node = node.getNext();
		}
		return tree_list;
	}

	/**
	 * iterates over all trees in the heap and returns the minimum node
	 */
	private void updateMin() {
		if (this.empty()) {
			this.setMin(null);
			return;
		}

		HeapNode cur_min = this.getLast();
		HeapNode node = this.getLast().getNext();
		while (node != this.getLast()) {
			if (node.getItem().getKey() < cur_min.getItem().getKey()) {
				cur_min = node;
			}
			node = node.getNext();
		}
		this.setMin(cur_min);
	}

	/**
	 *
	 * Return the number of elements in the heap
	 *
	 */
	public int size()
	{
		return this.size;
	}

	/**
	 *
	 * The method returns true if and only if the heap
	 * is empty.
	 *
	 */
	public boolean empty()
	{
		return this.size() == 0;
	}

	/**
	 *
	 * Return the number of trees in the heap.
	 *
	 */
	public int numTrees()
	{
		return Integer.bitCount(this.size());
		/**
		 * bitCount(n) return the number of 1's in the binary representation of n.
		 * This is equivalent to the number of tree in our binomial heap.
		 * The time complexity of this function is O(1) - https://stackoverflow.com/questions/44250311/java-big-o-of-bitcount
		 **/
	}

	public int highest_rank() {
		return (int) Math.floor(Math.log(this.size()) / Math.log(2));
	}

	/***
	 * pre: x.getRank() == y.getRank();
	 *
	 * Links two binomial heaps.
	 * Translating the pseudocode seen in class, for the circular list representation.
	 *
	 */
	public static HeapNode link(HeapNode x, HeapNode y) {
		if (x.getItem().getKey() > y.getItem().getKey()) { // if x > y, then x <--> y
			HeapNode tmp = x;
			x = y;
			y = tmp;
		}

		if (x.getChild() == null) {
			y.setNext(y);
		}
		else {
			y.setNext(x.getChild().getNext());
			x.getChild().setNext(y);
		}
		x.setChild(y);
		x.setRank(x.getRank() + 1); // rank increases by 1 when linking another tree
		y.setParent(x);	// y's parent is now x

		return x;
	}

	/**
	 * Class implementing a node in a Binomial Heap.
	 *
	 */
	public class HeapNode{
		public HeapItem item;
		public HeapNode child;
		public HeapNode next;
		public HeapNode parent;
		public int rank;

		public HeapNode() { // Default constructor
			this.item = null;
			this.child = null;
			this.next = null;
			this.parent = null;
		}

		public HeapNode(HeapItem item) { // constructor, initializes node with no connections, only an associated item
			this.item = item;
			this.child = null;
			this.next = null;
			this.parent = null;
		}

		/**
		 * Swaps items given two nodes
		 */
		private void swap(HeapNode node) {
			HeapItem tmp  = node.getItem();
			this.getItem().setNode(node);
			node.getItem().setNode(this);
			node.setItem(this.getItem());
			this.setItem(tmp);
		}

		public HeapItem getItem() {
			return this.item;
		}

		public HeapNode getChild() {
			return this.child;
		}

		public HeapNode getNext() {
			return this.next;
		}

		public HeapNode getParent() {
			return this.parent;
		}

		public int getRank() {
			return this.rank;
		}

		public void setItem(HeapItem item) {
			this.item = item;
		}

		public void setChild(HeapNode child) {
			this.child = child;
		}

		public void setNext(HeapNode next) {
			this.next = next;
		}

		public void setParent(HeapNode parent) {
			this.parent = parent;
		}

		public void setRank(int rank) {
			this.rank = rank;
		}
	}

	/**
	 * Class implementing an item in a Binomial Heap.
	 *
	 */
	public class HeapItem{
		public HeapNode node;
		public int key;
		public String info;

		public HeapItem() { // Default Constructor
			this.key = 0;
			this.info = null;
		}

		public HeapItem(int key, String info) { // Constructor initializes item with corresponding node
			this.key = key;
			this.info = info;
		}

		public HeapNode getNode() {
			return this.node;
		}

		public int getKey() {
			return this.key;
		}

		public String getInfo() {
			return this.info;
		}

		public void setNode(HeapNode node) {
			this.node = node;
		}

		public void setKey(int key) {
			this.key = key;
		}

		public void setInfo(String info) {
			this.info = info;
		}
	}
}
