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
}

pub fn minimax(node: (u32, i32, String), tree:Vec<(u32, i32, String)>, maximize: bool) -> i32 {
    let node_id = node.0;
    print!("{node_id}, {maximize}");
    if node.2.len() == 0 {
        return node.1
    } else {
        let children: Vec<u32> = node.2.split(",").into_iter().map(|x| x.parse().unwrap()).collect();
        let new_tree = tree.clone();
        let nodes = new_tree.iter().filter(|x| children.contains(&x.0)).collect::<Vec<&(u32, i32, String)>>();
        if maximize {
            let mut value = -9999;
            for node in nodes {
                value = max(value, minimax(node.clone(), tree.clone(), false))
            }
            return value
        }else {
            let mut value = 9999;
            
            for node in nodes {
                value = min(value, minimax(node.clone(), tree.clone(), true))
            }
            return value
        }
    }
}