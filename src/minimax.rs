use std::cmp::max;
use std::cmp::min;

#[cfg(test)]
mod test_minimax {
    use super::*;

    #[test]
    fn test_terminal_state() {
        let node = (0, 1, String::from(""));
        let tree = vec![node.clone()];
        assert_eq!(minimax(node.clone(), tree, true), 1)
    }

    #[test]
    fn test_two_steps() {
        let node1 = (0, 1, String::from("1"));
        let node2 = (1, -1, String::from(""));
        let tree = vec![node1.clone(), node2.clone()];
        assert_eq!(minimax(node1.clone(), tree, true), -1)
    }

    #[test]
    fn test_choice() {
        let node1 = (0, 0, String::from("1,2"));
        let node2 = (1, 1, String::from("3,4"));
        let node3 = (2, 1, String::from("5,6"));
        let node2a = (3, 1, String::from(""));
        let node2b = (4, 2, String::from(""));
        let node3a = (5, 3, String::from(""));
        let node3b = (6, 4, String::from(""));
        let tree = vec![
            node1.clone(),
            node2.clone(),
            node3.clone(),
            node2a.clone(),
            node2b.clone(),
            node3a.clone(),
            node3b.clone(),
        ];
        assert_eq!(minimax(node1.clone(), tree.clone(), true), 3);
        assert_eq!(negamax(node1.clone(), tree.clone()), 3);
        assert_eq!(negamax_alphabeta(&node1, &tree, 0, 100, 5), 3);
    }
}

// 30/09/2023
pub fn minimax(node: (u32, i32, String), tree: Vec<(u32, i32, String)>, maximize: bool) -> i32 {
    let node_id = node.0;
    print!("{node_id}, {maximize};\n");
    if node.2.len() == 0 {
        return node.1;
    } else {
        let children: Vec<u32> = node
            .2
            .split(",")
            .into_iter()
            .map(|x| x.parse().unwrap())
            .collect();
        let new_tree = tree.clone();
        let nodes = new_tree
            .iter()
            .filter(|x| children.contains(&x.0))
            .collect::<Vec<&(u32, i32, String)>>();
        if maximize {
            let mut value = -9999;
            for child_node in nodes {
                value = max(value, minimax(child_node.clone(), tree.clone(), false))
            }
            return value;
        } else {
            let mut value = 9999;

            for child_node in nodes {
                value = min(value, minimax(child_node.clone(), tree.clone(), true))
            }
            return value;
        }
    }
}

fn get_nodes_by_id(node_ids: Vec<u32>, tree: &Vec<(u32, i32, String)>) -> Vec<&(u32, i32, String)> {
    let nodes = tree
        .iter()
        .filter(|x| node_ids.contains(&x.0))
        .collect::<Vec<&(u32, i32, String)>>();
    nodes
}

fn string_to_idx(string: String) -> Vec<u32> {
    string
        .split(",")
        .into_iter()
        .map(|x| x.parse().unwrap())
        .collect()
}

// 01/09/2023
fn negamax(node: (u32, i32, String), tree: Vec<(u32, i32, String)>) -> i32 {
    if node.2.len() == 0 {
        return node.1;
    } else {
        let node_ids = string_to_idx(node.2);
        let new_tree = tree.clone();
        let nodes = get_nodes_by_id(node_ids, &new_tree);

        let mut value = -9999;
        for child_node in nodes {
            value = max(value, -negamax(child_node.clone(), tree.clone()))
        }
        return value;
    }
}

// 02/09/2023
fn negamax_alphabeta(
    node: &(u32, i32, String),
    tree: &Vec<(u32, i32, String)>,
    alpha: i32,
    beta: i32,
    depth: u32,
) -> i32 {
    let mut a = alpha;
    if depth == 0 || node.2.len() == 0 {
        return node.1;
    } else {
        let mut value = -9999;
        for child_node in get_nodes_by_id(string_to_idx(node.2.clone()), tree) {
            value = max(
                value,
                -negamax_alphabeta(child_node, tree, -beta, -a, depth - 1),
            );
            print!("value:{value},alpha:{a},beta:{beta},depth:{depth};\n");
            a = max(a, value);
            if a >= beta {
                break;
            }
        }
        return value;
    }
}
