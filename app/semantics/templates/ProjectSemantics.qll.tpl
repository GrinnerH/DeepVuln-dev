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

  predicate isModeledWrapperName(string n) {
    // __WRAPPER_NAME_OR_LIST__
  }

  class WrapperModeledBA extends BufferAccess {
    WrapperModeledBA() {
      exists(FunctionCall c |
        isModeledWrapperName(c.getTarget().getName()) and
        this = c
      )
    }

    override string getName() {
      result = this.(FunctionCall).getTarget().getName()
    }

    override Expr getBuffer(string bufferDesc, int accessType) {
      result = this.(FunctionCall).getArgument(__DEST_ARG__) and
      bufferDesc = "destination buffer" and
      accessType = 2
    }

    override Expr getSizeExpr() {
      // If __SIZE_ARG__ is -1 => treat as unbounded/unknown size (return none()).
      __SIZE_EXPR_BODY__
    }
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
