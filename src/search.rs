use std::collections::HashMap;

// 29/09/2023
pub fn binary_search(list: &[i32], target: i32) -> usize {
    let mut bottom: usize = 0;
    let mut top = list.len() - 1;
    while top != bottom {
        let mid = (bottom + top + 1) / 2;
        println!("{top}, {bottom}, {mid}");
        if list[mid] > target {
            top = mid - 1;
        } else {
            bottom = mid;
        }
    }
    return bottom;
}

#[cfg(test)]
mod test_binary_search {
    use super::*;

    #[test]
    fn single() {
        assert_eq!(binary_search(&[1i32], 1i32), 0)
    }

    #[test]
    fn two() {
        assert_eq!(binary_search(&[1i32, 2i32], 2i32), 1)
    }

    #[test]
    fn full() {
        assert_eq!(binary_search(&[1i32, 2i32, 3i32], 2i32), 1)
    }
}

#[cfg(test)]
mod test_graph_search {

    use super::*;

    #[test]
    fn find_node() {
        let tree: HashMap<&str, Vec<&str>> = HashMap::from([
            ("root", vec!["0", "1", "2"]),
            ("0", vec!["00", "01"]),
            ("1", vec!["10", "11", "12"]),
            ("2", vec!["20"]),
            ("00", vec!["001"]),
            ("01", vec!["010", "011"]),
            ("10", vec!["100", "101", "102", "103"]),
            ("11", vec!["110"]),
            ("12", vec![]),
            ("20", vec!["200", "201", "202", "203", "204"]),
            ("000", vec![]),
            ("001", vec![]),
            ("010", vec![]),
            ("011", vec![]),
            ("100", vec![]),
            ("101", vec![]),
            ("102", vec![]),
            ("103", vec![]),
            ("110", vec![]),
            ("200", vec![]),
            ("201", vec![]),
            ("202", vec![]),
            ("203", vec![]),
            ("204", vec![]),
        ]);

        assert_eq!(depth_first_search("201", "root", &tree), true)
    }
}

// 04/09/2023
fn depth_first_search(
    target_node: &str,
    current_node: &str,
    tree: &HashMap<&str, Vec<&str>>,
) -> bool {
    print!("current node {current_node};\n");
    let children = tree.get(current_node).unwrap();
    if current_node == target_node {
        return true;
    } else {
        let mut found = false;
        for child in children {
            if depth_first_search(target_node, child, tree) {
                found = true;
                break;
            }
        }
        return found;
    }
}
