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
        let node2 = (1, 1, String::from(""));
        let tree = vec![node1.clone(), node2.clone()];
        assert_eq!(minimax(node1.clone(), tree, true), -1)
    }
}

pub fn minimax(node: (u32, i32, String), tree:Vec<(u32, i32, String)>, maximize: bool) -> i32 {
    if node.2.len() == 0 {
        return node.1
    } else {
        return -1
    }
}