import { Link } from "react-router-dom";

export default function NotFound() {
  return (
    <div className="mx-auto max-w-lg px-4 py-16">
      <h1 className="mb-2 text-lg font-semibold">Page not found</h1>
      <p className="mb-4 text-sm text-slate-600">
        That address does not exist, or the resource belongs to another organisation.
      </p>
      <Link to="/search" className="text-sm text-slate-900 underline underline-offset-2">
        Back to search
      </Link>
    </div>
  );
}
