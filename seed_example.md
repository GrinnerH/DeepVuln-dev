{
  "cve_id": "...",
  "repo_path": "...",              # 本地源码路径
  "db_path": "...",                # codeql database path
  "vulnerable_file": "...",        # repo 相对路径
  "line_number": 123,              # 漏洞点行号（1-index）
  "baseline_query_path": "...",    # baseline .ql 或 .qls（可选；没有就跳过 baseline）
  "regression_query_path": "...",  # 回归 .ql 或 .qls（可选；没有就用 baseline 代替）
  "codeql_search_paths": ["..."],  # CodeQL pack/search-path（可选，但推荐）
}
