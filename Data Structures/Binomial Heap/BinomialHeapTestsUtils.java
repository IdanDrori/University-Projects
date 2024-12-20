
public class BinomialHeapTestsUtils {
    public static class MyAssertionError extends AssertionError {
        public MyAssertionError(String msg) {
            super(msg);
        }
    }
    public static <T> void assertEqual(T expected, T actual) throws AssertionError {
        if (expected != actual) {
            throw new MyAssertionError("expected: " + expected + " not equal to actual: " + actual);
        }
    }

    public static void assertTrue(boolean expression) throws AssertionError {
        if (!expression) {
            throw new MyAssertionError("");
        }
    }

    public static void assertIsHeap(BinomialHeap heap) {

    }

    public static void checkHeapNode(BinomialHeap.HeapNode node) {
        if (node == null) {
            return;
        }

        assertEqual(node, node.getItem().getNode());
        BinomialHeap.HeapNode child = node.getChild();
        int child_rank = 0;
        if (child != null) {
            BinomialHeap.HeapNode smallest_child = child.next;
            child = smallest_child;
            do {
                assertEqual(child_rank, child.rank);
                assertEqual(node, child.getParent());
                assertTrue(node.getItem().getKey() <= child.getItem().getKey());
                checkHeapNode(child);
                child_rank++;
                child = child.getNext();
            } while (child != smallest_child);
            assertEqual(node.getChild().getNext(), child);
        }
        assertEqual(node.rank, child_rank);
    }

    public static void assertHeap(BinomialHeap heap,
                                  BinomialHeap.HeapItem minItem,
                                  int size) {
        if (minItem != null) {
            assertEqual(minItem.key, heap.findMin().getKey());
            assertEqual(minItem.info, heap.findMin().getInfo());
        }
        assertEqual(minItem, heap.findMin());
        assertEqual(size, heap.size());
        assertEqual(size == 0, heap.empty());
        if (!heap.empty()) {
            assertEqual(heap.highest_rank(), heap.getLast().getRank());
            BinomialHeap.HeapNode node = heap.getLast();
            BinomialHeap.HeapNode smallest_node = node.next;
            do {
                if (node.next != node) {
                    if (node == heap.getLast()) {
                        assertTrue(node.getRank() > node.getNext().getRank());
                    }
                    else {
                        assertTrue(node.getRank() < node.getNext().getRank());
                    }
                }
                assertEqual(null, node.getParent());
                checkHeapNode(node);
                node = node.getNext();
            } while (node != smallest_node);
            assertEqual(null, heap.findMin().getNode().getParent());
        }
    }
}
