(define builtins
  '(domain
    named_range
    lift
    is_none
    make_tuple
    tuple_get
    reduce
    deref
    shift
    scan
    plus
    minus
    multiplies
    divides
    eq
    less
    greater
    if_
    not_
    and_
    or_
))

(define (as-list f x) (let ((g (lambda (x) (string-append (f x) ", "))))
  (string-append "[" (apply string-append (map g x)) "]")))

(define (sym->pystr x) (string-append "'" (symbol->string x) "'"))
(define (sym->py x) (string-append "ir.Sym(id=" (sym->pystr x) ")"))
(define (symref->py x) (string-append "ir.SymRef(id=" (sym->pystr x) ")"))

(define (tagged-list? x) (and (list? x) (symbol? (car x))))
(define (tagged-with? sym x) (and (tagged-list? x) (equal? sym (car x))))

(define (gt-program? x) (tagged-with? 'gt-program x))
(define (gt-fencil? x) (tagged-with? 'gt-fencil x))
(define (gt-function? x) (tagged-with? 'gt-function x))
(define (gt-stencil-closure? x) (tagged-with? 'gt-stencil-closure x))
(define (gt-lambda? x) (tagged-with? 'gt-lambda x))
(define (gt-offset? x) (tagged-with? 'gt-offset x))
(define (gt-axis? x) (tagged-with? 'gt-axis x))
(define (gt-none? x) (equal? 'gt-none x))
(define (gt-builtin? x) (member x builtins))

(define (gt-program->py expr)
  (string-append
    "ir.Program(function_definitions=" (as-list gt->py (cadr expr))
    ", fencil_definitions=" (as-list gt-fencil->py (cddr expr))
    ")\n"))

(define (gt-fencil->py expr)
  (string-append
    "ir.FencilDefinition(id=" (sym->pystr (cadr expr))
    ", params=" (as-list sym->py (caddr expr))
    ", closures=" (as-list gt-stencil-closure->py (cdddr expr))
    ")"))

(define (gt-function->py expr)
  (string-append
    "ir.FunctionDefinition(id=" (sym->pystr (cadr expr))
    ", params=" (as-list sym->py (caddr expr))
    ", expr=" (gt->py (cadddr expr))
    ")"))

(define (gt-stencil-closure->py expr)
  (string-append
    "ir.StencilClosure("
    "domain=" (gt->py (cadr expr))
    ", stencil=" (gt->py (caddr expr))
    ", output=" (symref->py (cadddr expr))
    ", inputs=" (as-list symref->py (cddddr expr))
    ")"))

(define (gt-builtin->py expr)
  (symref->py expr))

(define (gt-lambda->py expr)
  (string-append
    "ir.Lambda("
    "params=" (as-list sym->py (cadr expr))
    ", expr=" (gt->py (caddr expr))
    ")"))

(define (gt-call->py expr)
  (string-append
    "ir.FunCall("
    "fun=" (gt->py (car expr))
    ", args=" (as-list gt->py (cdr expr))
    ")"))

(define (gt-offset->py expr)
  (string-append
    "ir.OffsetLiteral("
    "value=" (let ((x (cadr expr)))
                  (cond ((integer? x) (number->string x))
                        ((string? x) (string-append "'" x "'"))
                        (else (error "unexpected type" x))))
    ")"))

(define (gt-axis->py expr)
  (string-append
    "ir.AxisLiteral("
    "value='" (cadr expr) "'"
    ")"))

(define (bool->py expr)
  (string-append
    "ir.BoolLiteral("
    "value=" (cond (expr "True") (else "False"))
    ")"))

(define (int->py expr)
  (string-append
    "ir.IntLiteral("
    "value=" (number->string expr)
    ")"))

(define (real->py expr)
  (string-append
    "ir.FloatLiteral("
    "value=" (number->string expr)
    ")"))

(define (gt-none->py expr) "ir.NoneLiteral()")


(define (gt->py expr)
  (cond ((boolean? expr) (bool->py expr))
        ((integer? expr) (int->py expr))
        ((real? expr) (real->py expr))
        ((gt-none? expr) (gt-none->py expr))
        ((symbol? expr) (symref->py expr))
        ((gt-program? expr) (gt-program->py expr))
        ((gt-fencil? expr) (gt-fencil->py expr))
        ((gt-function? expr) (gt-function->py expr))
        ((gt-stencil-closure? expr) (gt-stencil-closure->py expr))
        ((gt-builtin? expr) (gt-builtin->py expr))
        ((gt-lambda? expr) (gt-lambda->py expr))
        ((gt-offset? expr) (gt-offset->py expr))
        ((gt-axis? expr) (gt-axis->py expr))
        (else (gt-call->py expr))))

(define prgm (read))
(display (gt->py prgm))
(newline)
