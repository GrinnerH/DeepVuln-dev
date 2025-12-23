import cpp
import semmle.code.cpp.security.BufferAccess
import semmle.code.cpp.security.BufferWrite

module ProjectSemantics {

  /**
   * This file is generated deterministically from SemanticFacts.json.
   * Do NOT add query clauses (from/select) here.
   * Only provide project-specific semantic adapters (wrappers, capacity, guards).
   */

  // -------- Wrapper modeling (primary effectiveness lever) --------

  // __WRAPPER_CLASS_DEFS__

  predicate isModeledWrapperAccess(BufferAccess ba) {
    // __WRAPPER_ACCESS_PRED__
  }

  // -------- Capacity semantics (optional, for later expansion) --------

  predicate isProjectCapacityMacro(string name) {
    // __CAP_MACRO_OR_LIST__
  }

  predicate isProjectCapacityField(string name) {
    // __CAP_FIELD_OR_LIST__
  }

  // -------- Guard semantics (optional, for later expansion) --------

  predicate isProjectGuardExpr(Expr e) {
    // __GUARD_PATTERN_BODY__
    1 = 0
  }
}
