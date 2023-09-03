def resolve(clause_a, clause_b):
    new_clause = []
    for literal in clause_a:
        if ("not " + literal) not in clause_b:
            new_clause.append(literal)
    for literal in clause_b:
        if ("not " + literal) not in clause_a:
            new_clause.append(literal)
    return new_clause

def is_consistent(knowledge_base):
    while True:
        new_clauses = []
        for i in range(len(knowledge_base)):
            for j in range(i + 1, len(knowledge_base)):
                clause_a = knowledge_base[i]
                clause_b = knowledge_base[j]
                resolvents = resolve(clause_a, clause_b)
                if not resolvents:
                    return False  # Contradiction found, knowledge is inconsistent
                new_clauses.extend(resolvents)
        if all(new_clause in knowledge_base for new_clause in new_clauses):
            return True  # No new clauses to add, knowledge is consistent
        knowledge_base.extend(new_clauses)

if __name__ == "__main__":
    knowledge_base = [
        ["A", "B"],
        ["not B", "C"],
        ["not C"]
    ]

    is_consistent_result = is_consistent(knowledge_base)

    if is_consistent_result:
        print("Knowledge is consistent.")
    else:
        print("Knowledge is inconsistent. A contradiction was found.")